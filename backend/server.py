from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from dotenv import load_dotenv
from track import handle_video_tracking, BoundingBox, YoloDetector
import os
import json

# Loads variables from .env into the environment
load_dotenv()
app = FastAPI()

# Configure paths
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
MODEL_PATH = os.getenv("MODEL_PATH")
TEMP_DIR = os.getenv("TEMP_DIR")
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize detector
detector = YoloDetector(model_path=MODEL_PATH, confidence=0.4)


@app.post("/process-video")
async def process_video(video: UploadFile = File(...), bbox: str = Form(...)):
    if not video.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    try:
        # Parse the bbox JSON string to dict
        bbox_data = json.loads(bbox)
        # Validate bbox data
        bbox = BoundingBox(**bbox_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid bbox JSON format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid bbox data: {str(e)}")

    return await handle_video_tracking(video, bbox, detector, TEMP_DIR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
