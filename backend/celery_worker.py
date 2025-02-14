from celery import Celery, chain
import time
import redis
import os
from dotenv import load_dotenv

from track import BoundingBox, YoloDetector

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB = os.getenv("REDIS_DB")

# Configure Celery and Redis
celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Initialize detector
MODEL_PATH = os.getenv("MODEL_PATH")
detector = YoloDetector(model_path=MODEL_PATH, confidence=0.4)

TEMP_DIR = os.getenv("TEMP_DIR")
os.makedirs(TEMP_DIR, exist_ok=True)


@celery.task(bind=True)
def track_and_crop(self, input_path: str, bbox: BoundingBox):
    # """Step 1: Track and Crop video"""
    CELERY_STEP = "Tracking & Cropping"

    temp_output = input_path.replace("_input.mp4", "_output.mp4")
    temp_cropped = input_path.replace("_input.mp4", "_cropped.mp4")

    try:
        with open(input_path, "wb") as f:
            content = video.read()
            f.write(content)

        # Process video
        tracker = Tracker(iou_threshold=0.2)
        cap = cv2.VideoCapture(input_path)

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

        self.update_state(state="PROGRESS", meta={"step": CELERY_STEP, "progress": 0})

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
            self.update_state(
                state="PROGRESS",
                meta={
                    "step": CELERY_STEP,
                    "progress": int((processed_frames / total_frames) * 100),
                },
            )

        # Clean up
        cap.release()
        out.release()
        cropped_out.release()

        # Return the cropped video
        return temp_cropped

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary files
        for temp_file in [input_path, temp_output]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


@celery.task(bind=True)
def upscale_video(self, cropped_path: str):
    pass


@celery.task(bind=True)
def perform_ocr(self, upscaled_path: str):
    pass
