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
        workflow = chain(
            track_and_crop.s(input_path, bbox), upscale_video.s(), perform_ocr.s()
        )

        result = workflow.apply_async()
        logger.debug(f"Created chain with ID: {result.id}")

        # Try to get initial state
        initial_state = result.state
        logger.debug(f"Initial chain state: {initial_state}")

        # Store the task information
        task_metadata[task_id] = {
            "chain_id": result.id,
            "input_path": input_path,
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
    chain_id = metadata["chain_id"]
    logger.debug(f"Found chain_id: {chain_id}")

    # Now celery is defined, so we can use it
    chain_result = AsyncResult(chain_id, app=celery)
    logger.debug(f"Chain state: {chain_result.state}")
    logger.debug(f"Chain info: {chain_result.info}")

    # Traverse back through parent tasks
    current_task = chain_result
    while current_task and current_task.parent:
        current_task = current_task.parent
    logger.debug(f"Current task state: {current_task.state}")
    logger.debug(f"Current task info: {current_task.info}")

    # Ensure meta info is a dictionary
    result_meta = current_task.info or {}

    # Handle "PROGRESS" state
    if isinstance(result_meta, dict) and "step" in result_meta:
        response = {
            "task_id": task_id,
            "status": "PROGRESS",
            "current_task": result_meta.get("step", "Unknown"),
            "progress": result_meta.get("progress", 0),
        }
        logger.debug(f"Returning progress response: {response}")
        return response

    # Handle different states
    if chain_result.state == "PENDING":
        response = {
            "task_id": task_id,
            "status": "PENDING",
            "message": "Task is pending execution",
        }
        logger.debug(f"Returning pending response: {response}")
        return response

    if chain_result.state == "FAILURE":
        error_msg = str(chain_result.result)
        logger.error(f"Task failed: {error_msg}")
        return {"task_id": task_id, "status": "FAILURE", "error": error_msg}

    if chain_result.state == "SUCCESS":
        final_result = chain_result.get()
        task_metadata[task_id]["output_path"] = final_result
        response = {"task_id": task_id, "status": "SUCCESS", "output": final_result}
        logger.debug(f"Returning success response: {response}")
        return response

    # Default fallback
    response = {"task_id": task_id, "status": chain_result.state}
    logger.debug(f"Returning default response: {response}")
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
