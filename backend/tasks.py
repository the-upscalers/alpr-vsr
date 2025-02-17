from celery import Celery, states
import time
import os
import json
import cv2
import torch
from dotenv import load_dotenv

from track import BoundingBox, YoloDetector, Tracker
from upscale import load_realbasicvsr, load_video_to_tensor, convert_tensor_to_video
from ocr import perform_ocr_on_video, LicensePlateOCR

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "pyamqp://guest@rabbitmq//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

PROGRESS_STATE = "PROGRESS"

# Configure Celery and Redis
celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
)

# Initialize detector and tracker
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
detector = YoloDetector(model_path=YOLO_MODEL_PATH, confidence=0.4)
tracker = Tracker(iou_threshold=0.2)

# Load RealBasicVSR model
realBasicVSR = load_realbasicvsr()

# Load OCR model
ocr_model = LicensePlateOCR()

TEMP_DIR = os.getenv("TEMP_DIR")
os.makedirs(TEMP_DIR, exist_ok=True)


@celery.task(bind=True)
def run_pipeline(self, input_path: str, bbox: str):
    """Process input video through the pipeline"""
    CELERY_STEP = "Tracking & Cropping"

    self.update_state(state=PROGRESS_STATE, meta={"step": CELERY_STEP, "progress": 0})

    # Parse bounding box data
    try:
        bbox_data = json.loads(bbox)
        bbox = BoundingBox(**bbox_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid bbox JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Invalid bbox data: {str(e)}")

    temp_output = input_path.replace("_input.mp4", "_output.mp4")
    temp_cropped = input_path.replace("_input.mp4", "_cropped.mp4")

    try:
        # Process video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {input_path}")

        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            temp_output,
            fourcc,
            cap.get(cv2.CAP_PROP_FPS),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

        cropped_out = cv2.VideoWriter(
            temp_cropped,
            fourcc,
            cap.get(cv2.CAP_PROP_FPS),
            (bbox.x2 - bbox.x1, bbox.y2 - bbox.y1),
        )

        target_initialized = False
        target_box = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)

            if not target_initialized:
                target_initialized = tracker.initialize_target(
                    target_box, detections, frame
                )
                if not target_initialized:
                    continue

            if target_initialized:
                tracking_id, box = tracker.track(detections, frame)
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)

                    # Crop the plate region
                    cropped_plate = frame[y1:y2, x1:x2]
                    if cropped_plate.size > 0:
                        resized_plate = cv2.resize(
                            cropped_plate, (bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
                        )
                        cropped_out.write(resized_plate)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            out.write(frame)
            processed_frames += 1

            # Update progress every 10 frames or so
            if processed_frames % 10 == 0:
                self.update_state(
                    state=PROGRESS_STATE,
                    meta={
                        "step": CELERY_STEP,
                        "progress": int(((processed_frames / total_frames) * 100) / 3),
                    },
                )

        # Clean up
        cap.release()
        out.release()
        cropped_out.release()

    except Exception as e:
        self.update_state(state="FAILURE", meta={"step": CELERY_STEP, "error": str(e)})
        raise

    finally:
        # Clean up temporary files
        for temp_file in [input_path, temp_output]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Upscale the cropped video
    CELERY_STEP = "Upscaling"
    self.update_state(state=PROGRESS_STATE, meta={"step": CELERY_STEP, "progress": 33})
    output_path = temp_cropped.replace("_cropped.mp4", "_upscaled.mp4")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        video_tensor = load_video_to_tensor(temp_cropped)
        video_tensor = video_tensor.to(device)

        self.update_state(
            state=PROGRESS_STATE,
            meta={
                "step": CELERY_STEP,
                "progress": 45,
            },
        )

        realBasicVSR.eval()
        with torch.no_grad():
            sr_video_tensor = realBasicVSR(video_tensor)

        self.update_state(
            state=PROGRESS_STATE,
            meta={
                "step": CELERY_STEP,
                "progress": 60,
            },
        )

        convert_tensor_to_video(sr_video_tensor, output_path)

        self.update_state(
            state=PROGRESS_STATE, meta={"step": CELERY_STEP, "progress": 66}
        )

    except Exception as e:
        self.update_state(state="FAILURE", meta={"step": CELERY_STEP, "error": str(e)})
        raise

    # Perform OCR on the upscaled video
    CELERY_STEP = "Performing OCR"
    self.update_state(state=PROGRESS_STATE, meta={"step": CELERY_STEP, "progress": 0})
    output_path = output_path.replace("_upscaled.mp4", "_ocr.png")

    try:
        self.update_state(
            state=PROGRESS_STATE,
            meta={
                "step": CELERY_STEP,
                "progress": 50,
            },
        )

        perform_ocr_on_video(ocr_model, upscaled_path, output_path)

        self.update_state(
            state=PROGRESS_STATE,
            meta={
                "step": CELERY_STEP,
                "progress": 100,
            },
        )
    except Exception as e:
        self.update_state(state="FAILURE", meta={"step": CELERY_STEP, "error": str(e)})
        raise

    return output_path
