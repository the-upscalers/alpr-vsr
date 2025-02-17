from fastapi import UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import cv2
import numpy as np
import os

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Tracker:
    def __init__(self, max_features=500):
        self.object_tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=0.1,
            max_cosine_distance=0.6,
            nn_budget=None,
            embedder="mobilenet",
            half=True,
            bgr=True
        )
        # Initialize feature detector
        self.feature_detector = cv2.SIFT_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        self.target_track_id = None
        self.target_initialized = False
        self.target_keypoints = None
        self.target_descriptors = None
        
    def initialize_target(self, target_box, frame):
        """Initialize tracking using feature matching"""
        try:
            # Extract the target region
            x1, y1, x2, y2 = [int(coord) for coord in target_box]
            target_region = frame[y1:y2, x1:x2]
            
            # Convert to grayscale for feature detection
            gray_target = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)
            
            # Detect features in target region
            self.target_keypoints, self.target_descriptors = self.feature_detector.detectAndCompute(gray_target, None)
            
            if self.target_keypoints and len(self.target_keypoints) >= 4:
                self.target_initialized = True
                # Store original box dimensions for reference
                self.target_dimensions = (x2 - x1, y2 - y1)
                return True
                
            return False
            
        except Exception as e:
            print(f"Error initializing target: {e}")
            return False
            
    def find_best_match(self, detections, frame):
        """Find the best matching detection using feature matching"""
        if not self.target_initialized or self.target_descriptors is None:
            return None
            
        best_match = None
        max_good_matches = 0
        
        for detection in detections:
            try:
                bbox = detection[0]  # [x, y, w, h]
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                
                # Extract detection region
                detection_region = frame[y1:y2, x1:x2]
                if detection_region.size == 0:
                    continue
                    
                # Convert to grayscale
                gray_detection = cv2.cvtColor(detection_region, cv2.COLOR_BGR2GRAY)
                
                # Detect features in current detection
                det_keypoints, det_descriptors = self.feature_detector.detectAndCompute(gray_detection, None)
                
                if det_descriptors is None or len(det_keypoints) < 4:
                    continue
                    
                # Match features
                matches = self.matcher.match(self.target_descriptors, det_descriptors)
                
                # Filter good matches based on distance
                good_matches = [m for m in matches if m.distance < 0.7 * max(m.distance for m in matches)]
                
                if len(good_matches) > max_good_matches:
                    max_good_matches = len(good_matches)
                    best_match = detection
                    
            except Exception as e:
                print(f"Error processing detection: {e}")
                continue
                
        return best_match
        
    def track(self, detections, frame):
        """Track using a combination of feature matching and DeepSORT"""
        if not detections or not self.target_initialized:
            return None, None
            
        # Find best match using features
        best_match = self.find_best_match(detections, frame)
        if best_match is None:
            return None, None
            
        # Update DeepSORT with best match
        tracks = self.object_tracker.update_tracks([best_match], frame=frame)
        
        for track in tracks:
            if track.is_confirmed():
                ltrb = track.to_ltrb()
                # Validate box coordinates
                if len(ltrb) == 4 and all(isinstance(x, (int, float)) for x in ltrb):
                    height, width = frame.shape[:2]
                    l, t, r, b = [int(x) for x in ltrb]
                    l = max(0, min(l, width))
                    t = max(0, min(t, height))
                    r = max(0, min(r, width))
                    b = max(0, min(b, height))
                    return track.track_id, (l, t, r, b)
                    
        return None, None

    def draw_matches(self, frame1, frame2, kp1, kp2, matches):
        """Utility function to visualize matches for debugging"""
        img_matches = cv2.drawMatches(
            frame1, kp1,
            frame2, kp2,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return img_matches

class YoloDetector:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.classList = ["licence-plate"]
        self.confidence = confidence

    def detect(self, image):
        if image is None or not isinstance(image, np.ndarray):
            print("Invalid image input")
            return []
            
        results = self.model.predict(image, conf=self.confidence, verbose=False)
        result = results[0]
        detections = self.make_detections(result)
        return detections

    def make_detections(self, result):
        boxes = result.boxes
        detections = []
        
        if not hasattr(result, 'names'):
            print("No class names found in result")
            return detections
            
        for box in boxes:
            try:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Validate box dimensions
                if w <= 0 or h <= 0:
                    continue
                    
                class_number = int(box.cls[0])
                
                if class_number >= len(result.names) or \
                   result.names[class_number] not in self.classList:
                    continue
                    
                conf = float(box.conf[0])
                detections.append(([x1, y1, w, h], class_number, conf))
            except (IndexError, AttributeError) as e:
                print(f"Error processing detection box: {e}")
                continue
                
        return detections

def crop_license_plate(frame, bbox):
    """Utility function to crop license plate from frame"""
    if frame is None or bbox is None:
        return None
        
    try:
        l, t, r, b = bbox
        # Ensure coordinates are integers and within frame boundaries
        height, width = frame.shape[:2]
        l = max(0, min(int(l), width))
        t = max(0, min(int(t), height))
        r = max(0, min(int(r), width))
        b = max(0, min(int(b), height))
        
        if r <= l or b <= t:
            return None
            
        return frame[t:b, l:r]
    except Exception as e:
        print(f"Error cropping license plate: {e}")
        return None