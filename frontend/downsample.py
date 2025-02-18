import cv2


def downsample_video(input_path, output_path, scale_factor):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Use a more efficient codec (H.264)
    codec = cv2.VideoWriter_fourcc(*"X264")
    out = cv2.VideoWriter(output_path, codec, fps, (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Downsample using INTER_AREA for better quality & compression
        downsampled_frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        out.write(downsampled_frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    input_video = "videos/car-park-2.mp4"
    output_video = "videos/downsampled_car_park_2.mp4"
    scale_factor = 0.45

    downsample_video(input_video, output_video, scale_factor)
