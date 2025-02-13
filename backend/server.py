# server/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import cv2
import os
from typing import List
import tempfile
from track import Tracker, YoloDetector
import json

from dotenv import load_dotenv
import os

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

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

@app.post("/process-video")
async def process_video(video: UploadFile = File(...), bbox: str = Form(...)):
    if not video.filename.endswith(('.mp4', '.avi', '.mov')):
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
    
    # Save uploaded video temporarily
    temp_input = os.path.join(TEMP_DIR, f"input_{video.filename}")
    temp_output = os.path.join(TEMP_DIR, f"output_{video.filename}")
    temp_cropped = os.path.join(TEMP_DIR, f"cropped_{video.filename}")
    
    try:
        with open(temp_input, 'wb') as f:
            content = await video.read()
            f.write(content)
        
        # Process video
        tracker = Tracker(iou_threshold=0.2)
        cap = cv2.VideoCapture(temp_input)
        
        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, 
                            cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        cropped_out = cv2.VideoWriter(temp_cropped, fourcc, 
                                    cap.get(cv2.CAP_PROP_FPS),
                                    (bbox.x2 - bbox.x1, bbox.y2 - bbox.y1))
        
        target_initialized = False
        target_box = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)

            if not target_initialized:
                target_initialized = tracker.initialize_target(target_box, detections, frame)
                if not target_initialized:
                    continue

            if target_initialized:
                tracking_id, box = tracker.track(detections, frame)
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Crop the plate region
                    cropped_plate = frame[y1:y2, x1:x2]
                    if cropped_plate.size > 0:
                        resized_plate = cv2.resize(cropped_plate, 
                                                 (bbox.x2 - bbox.x1, bbox.y2 - bbox.y1))
                        cropped_out.write(resized_plate)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            out.write(frame)

        # Clean up
        cap.release()
        out.release()
        cropped_out.release()

        # Return the cropped video
        return FileResponse(temp_cropped, 
                          media_type="video/mp4", 
                          filename=f"cropped_{video.filename}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        pass
        # Clean up temporary files
        # for temp_file in [temp_input, temp_output, temp_cropped]:
        #     if os.path.exists(temp_file):
        #         os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)