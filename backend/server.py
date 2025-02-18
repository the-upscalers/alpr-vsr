import os
import json
import uuid
import logging

from track import BoundingBox
from dotenv import load_dotenv
from tasks import run_pipeline, celery
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from celery.result import AsyncResult

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Loads variables from .env into the environment
load_dotenv()
app = FastAPI()

# Configure paths
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
TEMP_DIR = os.getenv("TEMP_DIR")
os.makedirs(TEMP_DIR, exist_ok=True)

task_metadata = {}  # Store task metadata


@app.post("/process-video")
async def process_video(
    video: UploadFile = File(...), bbox: str = Form(...), frame_number: int = Form(...)
):
    """Process uploaded video"""
    logger.debug("Starting process_video endpoint")

    # Validate video format
    if not video.filename.endswith((".mp4")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    task_id = str(uuid.uuid4())
    logger.debug(f"Generated task_id: {task_id}")

    # Save uploaded video temporarily
    try:
        task_folder = os.path.join(TEMP_DIR, task_id)
        os.makedirs(task_folder, exist_ok=True)
        input_path = os.path.join(task_folder, task_id + "_input.mp4")

        with open(input_path, "wb") as f:
            content = await video.read()
            f.write(content)
        logger.debug(f"Saved video to: {input_path}")
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving video: {str(e)}")

    # Create and execute the chain
    try:
        result = run_pipeline.delay(input_path, bbox, frame_number)
        logger.debug(f"Created chain with ID: {result.id}")

        # Try to get initial state
        initial_state = result.state
        logger.debug(f"Initial chain state: {initial_state}")

        # Store the task information
        task_metadata[task_id] = {
            "celery_task_id": result.id,
            "input_path": input_path,
        }
        logger.debug(f"Stored task metadata: {task_metadata[task_id]}")

        return {"task_id": task_id}
    except Exception as e:
        logger.error(f"Error creating/executing chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Check progress of processing"""
    logger.debug(f"Checking status for task_id: {task_id}")

    if task_id not in task_metadata:
        logger.error(f"Invalid task_id: {task_id}")
        return {"error": "Invalid task ID"}

    metadata = task_metadata[task_id]
    celery_task_id = metadata.get("celery_task_id")

    task_result = AsyncResult(celery_task_id, app=celery)
    logger.debug(f"Task result: {task_result}")

    if task_result.status == "SUCCESS":
        task_metadata[task_id]["output_path"] = task_result.result
        return {"task_id": task_id, "status": "SUCCESS", "progress": 100}
    elif task_result.status == "FAILURE":
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "error": str(task_result.result),
        }
    elif task_result.status == "PENDING" or task_result.status == "STARTED":
        return {"task_id": task_id, "status": "PENDING", "progress": 0}
    elif task_result.status == "PROGRESS":
        progress = task_result.info.get("progress", 0)
        return {"task_id": task_id, "status": "PROGRESS", "progress": progress}

    return {"task_id": task_id, "status": task_result.status}


@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """Downloads the final processed file"""
    if task_id not in task_metadata:
        return {"error": "Invalid task ID"}

    final_output_path = task_metadata[task_id].get("output_path")
    if not final_output_path or not os.path.exists(final_output_path):
        return {"error": "File not available yet"}

    return FileResponse(
        final_output_path, media_type="text/plain", filename=f"{task_id}_ocr.png"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
