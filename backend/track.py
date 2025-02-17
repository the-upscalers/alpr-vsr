from fastapi import UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import cv2
import os


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Tracker:
    def __init__(self, iou_threshold=0.2):
        self.object_tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=0.1,
            max_cosine_distance=0.6,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None,
        )
        self.target_track_id = None
        self.target_initialized = False
        self.iou_threshold = iou_threshold

    def initialize_target(self, target_box, detections, frame):
        """Initialize tracking for the target license plate"""
        # Convert target_box from [left, top, right, bottom] to [left, top, width, height]
        target_w = target_box[2] - target_box[0]
        target_h = target_box[3] - target_box[1]
        target_bbox = [target_box[0], target_box[1], target_w, target_h]

        # Find the detection that best matches our target box
        best_iou = 0
        best_detection = None

        for detection in detections:
            det_bbox = detection[0]  # [x, y, w, h]
            iou = self.calculate_iou(target_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_detection = detection

        print(f"Best IoU: {best_iou}")
        if best_iou > self.iou_threshold:  # If we found a good match
            # Update tracks with only this detection
            tracks = self.object_tracker.update_tracks([best_detection], frame=frame)
            for track in tracks:
                if track.is_confirmed():
                    self.target_track_id = track.track_id
                    self.target_initialized = True
                    return True
        return False

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in [x, y, w, h] format"""
        # Convert to [x1, y1, x2, y2] format
        box1_x2 = box1[0] + box1[2]
        box1_y2 = box1[1] + box1[3]
        box2_x2 = box2[0] + box2[2]
        box2_y2 = box2[1] + box2[3]

        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def track(self, detections, frame):
        tracks = self.object_tracker.update_tracks(detections, frame=frame)

        if not self.target_initialized:
            return None, None

        # Only return the target track if found
        for track in tracks:
            if not track.is_confirmed():
                continue
            if track.track_id == self.target_track_id:
                ltrb = track.to_ltrb()
                return track.track_id, ltrb

        return None, None


class YoloDetector:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.classList = ["licence-plate"]
        self.confidence = confidence

    def detect(self, image):
        results = self.model.predict(image, conf=self.confidence, verbose=False)
        result = results[0]
        detections = self.make_detections(result)
        return detections

    def make_detections(self, result):
        boxes = result.boxes
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            class_number = int(box.cls[0])

            if result.names[class_number] not in self.classList:
                continue
            conf = box.conf[0]
            detections.append((([x1, y1, w, h]), class_number, conf))
        return detections
