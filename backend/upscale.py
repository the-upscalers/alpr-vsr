import os
import cv2
import torch
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
import numpy as np

from mmagic.models.editors.real_basicvsr.real_basicvsr import RealBasicVSR
from mmagic.models.editors.real_basicvsr.real_basicvsr_net import RealBasicVSRNet
from mmagic.models.editors.real_esrgan.unet_disc import (
    UNetDiscriminatorWithSpectralNorm,
)
from mmagic.models.losses.pixelwise_loss import L1Loss
from mmagic.models.losses.perceptual_loss import PerceptualLoss
from mmagic.models.losses.gan_loss import GANLoss
from mmagic.models.data_preprocessors.data_preprocessor import DataPreprocessor

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_BASIC_VSR_PATH = os.getenv("REAL_BASIC_VSR_PATH")
SPYNET_PATH = os.getenv("SPYNET_PATH")


def load_model_weights(model, checkpoint_path):
    """
    Load BasicVSR weights by removing the 'generator.' prefix from state dict keys.

    Args:
        model (BasicVSRNet): The model to load weights into
        checkpoint_path (str): Path to the checkpoint file

    Returns:
        BasicVSRNet: Model with loaded weights
    """
    # Load the state dict
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # If the state dict is wrapped in a larger dict, extract it
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    if "step_counter" in state_dict:
        del state_dict["step_counter"]

    # Load the modified state dict
    model.load_state_dict(state_dict, strict=True)
    return model


def load_video_to_tensor(video_path):
    """
    Load an mp4 video and transform it into a tensor with shape (N=1, T, C, H, W).
    Args:
        video_path (str): Path to the video file.
    Returns:
        torch.Tensor: A tensor of shape (N=1, T, C, H, W).
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = ToTensor()  # Converts HWC to CHW and scales pixel values to [0, 1]

    # Read video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV format) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Apply transformation
        frame_tensor = transform(frame_rgb)  # Converts to (C, H, W)
        frames.append(frame_tensor)

    cap.release()

    # Stack all frames along the time dimension
    video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)

    # Add a batch dimension
    video_tensor = video_tensor.unsqueeze(0)  # (N=1, T, C, H, W)
    return video_tensor


def convert_tensor_to_video(sr_video_tensor, output_path, fps=24):
    # Ensure tensor is in the format (T, H, W, C) and move to CPU
    sr_video_tensor = sr_video_tensor.squeeze(0)  # Remove batch dimension (N=1)
    sr_video_tensor = sr_video_tensor.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)

    # Convert pixel values to range [0, 255] for saving
    sr_video_tensor = (sr_video_tensor * 255).clip(0, 255).astype(np.uint8)

    # Get video dimensions
    T, H, W, C = sr_video_tensor.shape
    assert C == 3, "Video frames must have 3 channels (RGB/BGR)."

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for frame in sr_video_tensor:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")


def load_realbasicvsr():
    realBasicVSRNet = RealBasicVSRNet(
        mid_channels=64,
        num_propagation_blocks=20,
        num_cleaning_blocks=20,
        dynamic_refine_thres=255,  # change to 5 for test
        spynet_pretrained=SPYNET_PATH,
        is_fix_cleaning=False,
        is_sequential_cleaning=False,
    )

    discriminator = UNetDiscriminatorWithSpectralNorm(
        in_channels=3, mid_channels=64, skip_connection=True
    )

    pixel_loss = L1Loss(loss_weight=1.0, reduction="mean")
    cleaning_loss = L1Loss(loss_weight=1.0, reduction="mean")
    perceptual_loss = PerceptualLoss(
        layer_weights={
            "2": 0.1,
            "7": 0.1,
            "16": 1.0,
            "25": 1.0,
            "34": 1.0,
        },
        vgg_type="vgg19",
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False,
    )

    gan_loss = GANLoss(
        gan_type="vanilla", loss_weight=5e-2, real_label_val=1.0, fake_label_val=0
    )

    data_preprocessor = DataPreprocessor(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    realBasicVSR = RealBasicVSR(
        generator=realBasicVSRNet,
        discriminator=discriminator,
        pixel_loss=pixel_loss,
        cleaning_loss=cleaning_loss,
        perceptual_loss=perceptual_loss,
        gan_loss=gan_loss,
        is_use_sharpened_gt_in_pixel=True,
        is_use_sharpened_gt_in_percep=True,
        is_use_sharpened_gt_in_gan=False,
        is_use_ema=True,
        data_preprocessor=data_preprocessor,
    )
    realBasicVSR = load_model_weights(realBasicVSR, REAL_BASIC_VSR_PATH)
    realBasicVSR = realBasicVSR.to(device)

    return realBasicVSR


def main():
    video_path = "temp/cropped/cropped_number_plate_1.mp4"
    output_path = "temp/output/output_video_1_upscaled.mp4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    realBasicVSR = load_realbasicvsr()
    video_tensor = load_video_to_tensor(video_path)
    video_tensor = video_tensor.to(device)

    realBasicVSR.eval()
    with torch.no_grad():
        sr_video_tensor = realBasicVSR(video_tensor)

    convert_tensor_to_video(sr_video_tensor, output_path)


if __name__ == "__main__":
    main()
