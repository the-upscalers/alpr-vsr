from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import cv2
import time

class BoundingBoxDrawer:
    def __init__(self):
        self.drawing = False
        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1
        self.box_drawn = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x = x
            self.start_y = y
            self.end_x = x
            self.end_y = y
            self.box_drawn = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_x = x
                self.end_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_x = x
            self.end_y = y
            self.box_drawn = True

    def get_box(self):
        if not self.box_drawn:
            return None
        return [
            min(self.start_x, self.end_x),
            min(self.start_y, self.end_y),
            max(self.start_x, self.end_x),
            max(self.start_y, self.end_y)
        ]

    def draw_current_box(self, frame):
        if self.start_x != -1 and self.start_y != -1:
            cv2.rectangle(frame,
                          (self.start_x, self.start_y),
                          (self.end_x, self.end_y),
                          (255, 0, 0), 2)

class Tracker:
    def __init__(self):
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
            today=None
        )
        self.target_track_id = None
        self.target_initialized = False

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
        if best_iou > 0.3:  # If we found a good match
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

def main():
    MODEL_PATH = "models/YOLOv11.pt"
    VIDEO_PATH = "videos/parked-cars.mp4"
    OUTPUT_PATH = "output/tracked_video.mp4"

    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.4)
    tracker = Tracker()
    box_drawer = BoundingBoxDrawer()

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Create window and set mouse callback
    window_name = "Select License Plate"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, box_drawer.mouse_callback)

    # Selection phase
    print("Draw a box around the license plate you want to track.")
    print("Use 'a' to move backward, 'd' to move forward, 'p' to play/pause, and 'q' to quit.")
    print("Press SPACE when satisfied with the selection.")

    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    selection_frame = frame.copy()
    target_box = None
    paused = True
    current_frame = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video.")
                break
            selection_frame = frame.copy()
            current_frame += 1

        display_frame = selection_frame.copy()
        box_drawer.draw_current_box(display_frame)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            return
        elif key == ord(' '):
            if box_drawer.box_drawn:
                target_box = box_drawer.get_box()
                break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('a'):
            current_frame = max(0, current_frame - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                selection_frame = frame.copy()
        elif key == ord('d'):
            current_frame += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                selection_frame = frame.copy()

    # Tracking phase
    cv2.destroyWindow(window_name)

    # Output video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    target_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        if not target_initialized:
            target_initialized = tracker.initialize_target(target_box, detections, frame)
            if not target_initialized:
                print("Waiting for target initialization...")
                out.write(frame)
                continue

        if target_initialized:
            tracking_id, box = tracker.track(detections, frame)
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Target ID: {tracking_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Tracking completed. Output video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()