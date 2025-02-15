import torch
from dotenv import load_dotenv
import os

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

    del state_dict["step_counter"]

    # Load the modified state dict
    model.load_state_dict(state_dict, strict=True)
    return model


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

print(realBasicVSR)
