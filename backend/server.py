import os
import json
import uuid
import logging

from track import BoundingBox
from dotenv import load_dotenv
from tasks import track_and_crop, upscale_video, perform_ocr, celery
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from celery import chain
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
async def process_video(video: UploadFile = File(...), bbox: str = Form(...)):
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
        result = chain(
            track_and_crop.s(input_path, bbox), upscale_video.s(), perform_ocr.s()
        ).apply_async()
        logger.debug(f"Created chain with ID: {result.id}")

        # Try to get initial state
        initial_state = result.state
        logger.debug(f"Initial chain state: {initial_state}")

        # Store the task information
        task_metadata[task_id] = {
            "input_path": input_path,
            "task_ids": [result.parent.parent.id, result.parent.id, result.id],
            "task_sequence": ["track_and_crop", "upscale_video", "perform_ocr"],
            "current_task_index": 0,
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
    task_ids = [tid for tid in metadata.get("task_ids", []) if tid]
    task_sequence = metadata.get("task_sequence", [])

    task_states = {}
    total_progress = 0
    num_tasks = len(task_ids)

    if num_tasks == 0:
        return {"error": "No tasks found for this task ID"}

    for idx, id in enumerate(task_ids):
        task_result = AsyncResult(id, app=celery)
        task_states[task_sequence[idx]] = task_result.status

        if task_result.status == "SUCCESS":
            total_progress += 100 / num_tasks
        elif task_result.status == "PROGRESS" or task_result.status == "PENDING":
            step_progress = (
                task_result.info.get("progress", 0)
                if isinstance(task_result.info, dict)
                else 0
            )
            total_progress += step_progress / num_tasks
        elif task_result.status == "FAILURE":
            return {
                "task_id": task_id,
                "status": "FAILURE",
                "error": str(task_result.result),
                "task_states": task_states,
            }

    progress = int(total_progress)
    logger.debug(f"Task progress: {progress}%")

    response = {
        "task_id": task_id,
        "progress": progress,
        "task_states": task_states,
        "status": "PROGRESS" if progress < 100 else "SUCCESS",
    }
    logger.debug(f"Returning task status response: {response}")
    return response


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
