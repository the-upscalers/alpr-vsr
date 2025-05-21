from fastapi import UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import argparse
from pathlib import Path
import random

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class MultiTracker:
    def __init__(self, max_features=500, max_age=30, similarity_threshold=0.7):
        self.object_tracker = DeepSort(
            max_age=max_age,
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
        self.similarity_threshold = similarity_threshold
        
        # Dict to store active tracks: {track_id: (keypoints, descriptors, dimensions, last_seen_frame)}
        self.active_tracks = {}
        self.frame_count = 0
        
        # Dict to store track colors for visualization
        self.track_colors = {}
        
    def update(self, detections, frame):
        """Update all tracks with new detections"""
        self.frame_count += 1
        if not detections:
            return []
            
        # Update DeepSORT with all detections
        tracks = self.object_tracker.update_tracks(detections, frame=frame)
        
        result_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            ltrb = track.to_ltrb()
            # Validate box coordinates
            if len(ltrb) != 4 or not all(isinstance(x, (int, float)) for x in ltrb):
                continue
                
            height, width = frame.shape[:2]
            l, t, r, b = [int(x) for x in ltrb]
            l = max(0, min(l, width))
            t = max(0, min(t, height))
            r = max(0, min(r, width))
            b = max(0, min(b, height))
            
            track_id = track.track_id
            
            # Generate random color for new tracks
            if track_id not in self.track_colors:
                self.track_colors[track_id] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
            
            # Extract features for new or updated tracks
            if track_id not in self.active_tracks or self.frame_count % 10 == 0:
                # Extract region and features
                track_region = frame[t:b, l:r]
                if track_region.size == 0:
                    continue
                    
                try:
                    # Convert to grayscale for feature detection
                    gray_region = cv2.cvtColor(track_region, cv2.COLOR_BGR2GRAY)
                    
                    # Detect features
                    keypoints, descriptors = self.feature_detector.detectAndCompute(gray_region, None)
                    
                    if keypoints and len(keypoints) >= 4 and descriptors is not None:
                        # Store or update track information
                        self.active_tracks[track_id] = (keypoints, descriptors, (r-l, b-t), self.frame_count)
                except Exception as e:
                    print(f"Error extracting features for track {track_id}: {e}")
                    continue
            
            # Add to result
            result_tracks.append((track_id, (l, t, r, b), self.track_colors[track_id]))
            
        # Clean up old tracks
        self._clean_old_tracks()
        
        return result_tracks

    def _clean_old_tracks(self, max_frames_missing=60):
        """Remove tracks that haven't been seen for a while"""
        current_ids = list(self.active_tracks.keys())
        for track_id in current_ids:
            if self.frame_count - self.active_tracks[track_id][3] > max_frames_missing:
                del self.active_tracks[track_id]

class TextDetector:
    def __init__(self, model_path, confidence=0.3, classes=None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        # If classes not provided, include all classes from the model
        self.classes = classes
        
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
                class_name = result.names[class_number]
                
                # Filter by class if classes are specified
                if self.classes is not None and class_name not in self.classes:
                    continue
                    
                conf = float(box.conf[0])
                detections.append(([x1, y1, w, h], class_number, conf))
            except (IndexError, AttributeError) as e:
                print(f"Error processing detection box: {e}")
                continue
                
        return detections

def crop_text_region(frame, bbox, target_size=(256, 256)):
    """
    Utility function to crop a fixed-size region centered on the number plate
    
    Args:
        frame: Input frame
        bbox: Bounding box coordinates (l, t, r, b)
        target_size: Target size for output (width, height)
    
    Returns:
        Fixed-size region containing the number plate
    """
    if frame is None or bbox is None:
        return None
        
    try:
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Convert bbox coordinates to integers and ensure they're within frame boundaries
        l, t, r, b = bbox
        l = max(0, min(int(l), width))
        t = max(0, min(int(t), height))
        r = max(0, min(int(r), width))
        b = max(0, min(int(b), height))
        
        if r <= l or b <= t:
            return None
        
        # Calculate the center of the number plate
        center_x = (l + r) // 2
        center_y = (t + b) // 2
        
        # Calculate the target crop coordinates
        target_width, target_height = target_size
        half_width = target_width // 2
        half_height = target_height // 2
        
        # Calculate the crop coordinates centered on the plate
        crop_left = max(0, center_x - half_width)
        crop_top = max(0, center_y - half_height)
        
        # Adjust if the crop would go beyond the frame boundaries
        if crop_left + target_width > width:
            crop_left = max(0, width - target_width)
        if crop_top + target_height > height:
            crop_top = max(0, height - target_height)
        
        # Extract the fixed-size crop
        cropped = frame[crop_top:crop_top+target_height, crop_left:crop_left+target_width]
        
        # Check if we got a complete crop with the correct dimensions
        crop_height, crop_width = cropped.shape[:2]
        if crop_width != target_width or crop_height != target_height:
            # If not, create a black canvas of the target size and place the partial crop
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            # Calculate where to place the partial crop
            place_x = 0
            place_y = 0
            canvas[place_y:place_y+crop_height, place_x:place_x+crop_width] = cropped
            cropped = canvas
        
        return cropped
    except Exception as e:
        print(f"Error cropping text region: {e}")
        # Return a black image if an error occurs
        return np.zeros(target_size + (3,), dtype=np.uint8)

def draw_tracks(frame, tracks, show_ids=True, box_thickness=2, font_scale=0.5):
    """Draw bounding boxes and track IDs on frame"""
    result_frame = frame.copy()
    
    for track_id, bbox, color in tracks:
        l, t, r, b = bbox
        
        # Draw bounding box
        cv2.rectangle(result_frame, (l, t), (r, b), color, box_thickness)
        
        if show_ids:
            # Draw track ID
            text = f"ID: {track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            
            # Draw background rectangle for text
            cv2.rectangle(result_frame, 
                         (l, t - text_size[1] - 10), 
                         (l + text_size[0] + 10, t), 
                         color, 
                         -1)
            
            # Draw text
            cv2.putText(result_frame, 
                       text, 
                       (l + 5, t - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, 
                       (255, 255, 255), 
                       2)
    
    return result_frame

def process_video(video_path, model_path, output_folder, confidence=0.3, save_freq=10, 
                  classes=None, max_missing_frames=30, save_crops=True, visualize=True,
                  max_frames_per_track=None):
    """
    Process a video file to detect, track, and save text regions
    
    Args:
        video_path: Path to the input video
        model_path: Path to the YOLO model
        output_folder: Folder to save cropped text images and output video
        confidence: Detection confidence threshold
        save_freq: Save detected text every N frames
        classes: List of class names to detect (None for all classes in model)
        max_missing_frames: Remove tracks after this many frames of absence
        save_crops: Whether to save individual cropped images
        visualize: Whether to create visualization video
        max_frames_per_track: Maximum number of frames to save per track (None for unlimited)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize detector and tracker
    detector = TextDetector(model_path, confidence, classes)
    tracker = MultiTracker(max_age=max_missing_frames)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saved_tracks = set()  # Keep track of saved instances
    
    # Dict to track when we last saved an image for each track_id
    last_saved = {}
    
    # Dict to track how many frames we've saved for each track_id
    saved_frame_count = {}
    
    # Get video name for folder organization
    video_name = Path(video_path).stem
    
    # Create video-specific output folders
    crops_base_folder = os.path.join(output_folder, f"{video_name}_crops")
    os.makedirs(crops_base_folder, exist_ok=True)
    
    # Initialize video writer
    if visualize:
        visualization_path = os.path.join(output_folder, f"{video_name}_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(visualization_path, fourcc, fps, (width, height))
    
    # Initialize montage video (for crops)
    montage_path = os.path.join(output_folder, f"{video_name}_text_montage.mp4")
    montage_writer = None
    montage_frame = None
    montage_size = (1280, 720)  # Size of the montage frame
    max_crops_per_row = 4
    crop_display_size = (320, 180)  # Size of each crop in the montage
    
    print(f"Processing video: {video_path}")
    print(f"Saving output to: {output_folder}")
    
    # Store the latest crops for each track
    latest_crops = {}
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Detect text regions
        detections = detector.detect(frame)
        
        # Update tracks
        tracks = tracker.update(detections, frame)
        
        # Process each track
        for track_id, bbox, color in tracks:
            # Crop the text region
            cropped = crop_text_region(frame, bbox)
            if cropped is None:
                continue
                
            # Store the latest crop for this track
            latest_crops[track_id] = (cropped, color)
            
            # Save cropped image at regular intervals or when first detected
            if save_crops:
                # Initialize counter for this track if not exists
                if track_id not in saved_frame_count:
                    saved_frame_count[track_id] = 0
                
                # Check if we've reached the maximum frames for this track
                if max_frames_per_track is not None and saved_frame_count[track_id] >= max_frames_per_track:
                    continue
                
                time_to_save = track_id not in last_saved or \
                               frame_count - last_saved.get(track_id, 0) >= save_freq
                
                if time_to_save:
                    # Create a track-specific folder
                    track_folder = os.path.join(crops_base_folder, f"track_{track_id}")
                    os.makedirs(track_folder, exist_ok=True)
                    
                    # Create a filename with timestamp and track ID
                    timestamp = frame_count / fps
                    filename = f"{video_name}_track{track_id}_frame{frame_count}_time{timestamp:.2f}.jpg"
                    output_path = os.path.join(track_folder, filename)
                    
                    # Save the cropped image
                    cv2.imwrite(output_path, cropped)
                    last_saved[track_id] = frame_count
                    saved_tracks.add(track_id)
                    saved_frame_count[track_id] += 1
        
        # Create montage of currently tracked text regions
        if len(latest_crops) > 0:
            # Calculate layout
            num_crops = len(latest_crops)
            num_rows = (num_crops + max_crops_per_row - 1) // max_crops_per_row
            
            # Create blank montage frame
            montage_frame = np.zeros((montage_size[1], montage_size[0], 3), dtype=np.uint8)
            
            # Add crops to montage
            i = 0
            for track_id, (crop, color) in latest_crops.items():
                row = i // max_crops_per_row
                col = i % max_crops_per_row
                
                # Resize crop to display size
                resized_crop = cv2.resize(crop, crop_display_size)
                h, w = resized_crop.shape[:2]
                
                # Calculate position in montage
                x = col * crop_display_size[0]
                y = row * crop_display_size[1]
                
                # Check if we're still within montage bounds
                if y + h <= montage_size[1] and x + w <= montage_size[0]:
                    # Place crop in montage
                    montage_frame[y:y+h, x:x+w] = resized_crop
                    
                    # Add ID text
                    cv2.putText(montage_frame, 
                               f"ID: {track_id}", 
                               (x + 5, y + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, 
                               color, 
                               2)
                    
                    # Add frame count for this track
                    if track_id in saved_frame_count:
                        count_text = f"Frames: {saved_frame_count[track_id]}"
                        if max_frames_per_track is not None:
                            count_text += f"/{max_frames_per_track}"
                        cv2.putText(montage_frame, 
                                  count_text, 
                                  (x + 5, y + 45), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, 
                                  color, 
                                  1)
                
                i += 1
            
            # Initialize montage writer if not already done
            if montage_writer is None:
                montage_writer = cv2.VideoWriter(montage_path, fourcc, fps, montage_size)
            
            # Write frame to montage video
            montage_writer.write(montage_frame)
        
        # Visualize tracks on original frame
        if visualize:
            viz_frame = draw_tracks(frame, tracks)
            
            # Add frame counter
            cv2.putText(viz_frame, 
                       f"Frame: {frame_count}", 
                       (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (0, 255, 0), 
                       2)
            
            # Write to video
            video_writer.write(viz_frame)
        
        # Display progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, {len(saved_tracks)} unique text regions found")
    
    # Clean up
    cap.release()
    if visualize:
        video_writer.release()
    if montage_writer is not None:
        montage_writer.release()
    
    print(f"Finished processing. Found {len(saved_tracks)} unique text regions.")
    print(f"Visualization saved to: {os.path.join(output_folder, f'{video_name}_tracked.mp4')}")
    print(f"Text montage saved to: {os.path.join(output_folder, f'{video_name}_text_montage.mp4')}")
    
    return len(saved_tracks)

def main():
    parser = argparse.ArgumentParser(description="Detect and track text in videos")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--output", default="text_output", help="Output folder for videos and optionally crops")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--save-freq", type=int, default=2, help="Save text crops every N frames")
    parser.add_argument("--classes", nargs='+', default=None, help="Classes to detect (default: all)")
    parser.add_argument("--save-crops", action="store_true", help="Save individual cropped images")
    parser.add_argument("--no-visualization", dest="visualize", action="store_false", help="Do not create visualization video")
    parser.add_argument("--max-frames-per-track", type=int, default=None, help="Maximum number of frames to save per track (None for unlimited)")
    parser.set_defaults(save_crops=False, visualize=True)
    
    args = parser.parse_args()

    videos = [
        "downloads/cctv_9.mp4",
        "downloads/cctv_10.mp4",
        "downloads/cctv_11.mp4",
        "downloads/cctv_12.mp4",
        "downloads/cctv_13.mp4",
        "downloads/cctv_14.mp4",
    ]

    for video in videos:
        # Check if input is a directory or a single file
        if os.path.isdir(video):
            total_regions = 0
            for video_file in os.listdir(video):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(video, video_file)
                    regions = process_video(
                        video_path, 
                        args.model, 
                        args.output, 
                        args.conf, 
                        args.save_freq,
                        args.classes,
                        save_crops=args.save_crops,
                        visualize=args.visualize,
                        max_frames_per_track=args.max_frames_per_track
                    )
                    total_regions += regions if regions else 0
            print(f"Total unique text regions found across all videos: {total_regions}")
        else:
            process_video(
                video, 
                args.model, 
                args.output, 
                args.conf, 
                args.save_freq,
                args.classes,
                save_crops=args.save_crops,
                visualize=args.visualize,
                max_frames_per_track=args.max_frames_per_track
            )

if __name__ == "__main__":
    main()

# python track-text.py --video "../frontend/videos/cctv.mp4" --model "models/YOLOv11.pt" --save-crops --conf 0.5