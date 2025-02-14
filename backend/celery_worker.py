from celery import Celery, chain
import time
import redis
import os
from dotenv import load_dotenv

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB = os.getenv("REDIS_DB")

celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

TEMP_DIR = os.getenv("TEMP_DIR")
os.makedirs(TEMP_DIR, exist_ok=True)


@celery.task(bind=True)
def track_and_crop(self, input_path: str):
    """Step 1: Track and Crop video"""
    output_path = input_path.replace("_input.mp4", "_cropped.mp4")

    # Simulating processing time
    for i in range(1, 4):
        time.sleep(2)
        self.update_state(
            state="PROGRESS", meta={"step": "tracking & cropping", "progress": i * 25}
        )

    # Fake output (replace with real video processing)
    with open(output_path, "wb") as f:
        f.write(b"Fake cropped video data")

    return output_path
