import os
import json
import uuid

from track import BoundingBox
from dotenv import load_dotenv
from celery_worker import track_and_crop, upscale_video, perform_ocr
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from celery import chain

# Loads variables from .env into the environment
load_dotenv()
app = FastAPI()

# Configure paths
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
TEMP_DIR = os.getenv("TEMP_DIR")

task_metadata = {}  # Store task metadata


@app.post("/process-video")
async def process_video(video: UploadFile = File(...), bbox: str = Form(...)):
    """Process uploaded video"""

    # Validate video format
    if not video.filename.endswith((".mp4")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    task_id = str(uuid.uuid4())  # Generate unique task ID

    # Save uploaded video temporarily
    try:
        task_folder = os.path.join(TEMP_DIR, task_id)
        os.makedirs(task_folder, exist_ok=True)
        input_path = os.path.join(task_folder, task_id + "_input.mp4")

        with open(input_path, "wb") as f:
            content = await video.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving video: {str(e)}")

    result = chain(
        track_and_crop.s(input_path, bbox) | upscale_video.s() | perform_ocr.s()
    )()

    task_metadata[task_id] = {"celery_task_id": result.id, "input_path": input_path}
    return {"task_id": task_id}


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Check progress of processing"""
    if task_id not in task_metadata:
        return {"error": "Invalid task ID"}

    celery_task_id = task_metadata[task_id]["celery_task_id"]
    task = track_and_crop.AsyncResult(celery_task_id)

    if task.state == "PROGRESS":
        return {"task_id": task_id, "status": task.state, "progress": task.info}

    if task.state == "SUCCESS":
        return {"task_id": task_id, "status": "SUCCESS", "output": task.result}

    return {"task_id": task_id, "status": task.state}


@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """Downloads the final processed file"""
    if task_id not in task_metadata:
        return {"error": "Invalid task ID"}

    final_output_path = task_metadata[task_id].get("output_path")
    if not final_output_path or not os.path.exists(final_output_path):
        return {"error": "File not available yet"}

    return FileResponse(
        final_output_path, media_type="text/plain", filename=f"{task_id}_ocr.txt"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
