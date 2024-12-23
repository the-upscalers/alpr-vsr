from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                          QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import sys
import cv2
from track import Tracker, BoundingBoxDrawer, YoloDetector

class TrackingThread(QThread):
    # Define signals for communication with main thread
    frame_ready = pyqtSignal(object)  # For processed frame
    tracking_complete = pyqtSignal()   # Signal when tracking is done
    progress_update = pyqtSignal(str)  # For status updates

    def __init__(self, video_path, target_box, detector, tracker, output_path, cropped_output_path):
        super().__init__()
        self.video_path = video_path
        self.target_box = target_box
        self.detector = detector
        self.tracker = tracker
        self.output_path = output_path
        self.cropped_output_path = cropped_output_path
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, 
                            cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cropped_out = cv2.VideoWriter(self.cropped_output_path, fourcc, 
                            cap.get(cv2.CAP_PROP_FPS),
                            (self.target_box[2] - self.target_box[0], self.target_box[3] - self.target_box[1]))

        target_initialized = False

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)

            if not target_initialized:
                target_initialized = self.tracker.initialize_target(
                    self.target_box, detections, frame)
                if not target_initialized:
                    self.progress_update.emit("Waiting for target initialization...")
                    continue

            if target_initialized:
                tracking_id, box = self.tracker.track(detections, frame)
                if box is not None:
                    # Draw tracking box
                    x1, y1, x2, y2 = map(int, box)

                    # Crop the number plate region
                    cropped_plate = frame[y1:y2, x1:x2]
                    if cropped_plate.size > 0:  # Ensure the crop is valid
                        resized_plate = cv2.resize(cropped_plate, 
                                                (self.target_box[2] - self.target_box[0], self.target_box[3] - self.target_box[1]))
                        cropped_out.write(resized_plate)


                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Emit the processed frame to update UI
            self.frame_ready.emit(frame)
            
            # Write frame to output video
            out.write(frame)

        cap.release()
        out.release()
        cropped_out.release()
        print(f"Tracking completed. Output video saved to {self.output_path}")
        print(f"Cropped number plate video saved to {self.cropped_output_path}")
        self.tracking_complete.emit()

    def stop(self):
        self.is_running = False

class VideoPlayerWindow(QMainWindow):
    def __init__(self, video_path, model_path, output_path, cropped_output_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.cropped_output_path = cropped_output_path
        self.setup_tracking()
        self.setup_ui()
        self.setup_video()

    def setup_ui(self):
        self.setWindowTitle("License Plate Tracker")
        self.setMinimumSize(800, 600)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create video display container
        video_container = QWidget()
        video_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        # Create video display
        self.video_label = QLabel()
        self.video_label.setFrameStyle(QFrame.Shape.Box)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(640, 480)  # Set minimum size
        video_layout.addWidget(self.video_label)
        layout.addWidget(video_container)

        # Create controls widget
        controls_widget = QWidget()
        controls_widget.setFixedHeight(60)  # Fixed height for controls
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Create control buttons
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setFixedWidth(80)
        
        self.prev_frame_button = QPushButton("◀")
        self.prev_frame_button.setFixedWidth(50)
        
        self.next_frame_button = QPushButton("▶")
        self.next_frame_button.setFixedWidth(50)
        
        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.setFixedWidth(150)

        # Style buttons
        self.style_buttons()

        # Add buttons to controls layout
        controls_layout.addStretch()
        controls_layout.addWidget(self.prev_frame_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.next_frame_button)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.confirm_button)
        controls_layout.addStretch()

        # Add controls to main layout
        layout.addWidget(controls_widget)

        # Connect button signals
        self.play_button.clicked.connect(self.toggle_play)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.confirm_button.clicked.connect(self.confirm_selection)

        # Create status bar for instructions
        self.statusBar().showMessage("Draw a box around the license plate you want to track")

    def style_buttons(self):
        # Style all buttons
        style = """
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                min-width: 80px;
                color: #333;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
        """
        for button in [self.play_button, self.prev_frame_button, 
                      self.next_frame_button, self.confirm_button]:
            button.setStyleSheet(style)

    def setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize video state
        self.is_playing = False
        self.current_frame = None
        self.box_drawer = BoundingBoxDrawer()
        
        # Read first frame
        self.read_frame()

    def setup_tracking(self):
        self.detector = YoloDetector(model_path=self.model_path, confidence=0.4)
        self.tracker = Tracker(iou_threshold=0.2)
        self.tracking_initialized = False

    def toggle_play(self):
        self.is_playing = self.play_button.isChecked()
        if self.is_playing:
            self.timer.start(int(1000 / 30))  # 30 fps
            self.play_button.setText("Pause")
        else:
            self.timer.stop()
            self.play_button.setText("Play")

    def read_frame(self):
        ret, self.current_frame = self.cap.read()
        if ret:
            self.display_frame()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, self.current_frame = self.cap.read()

    def display_frame(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        
        # Draw current box if being drawn
        if not self.tracking_initialized:
            self.box_drawer.draw_current_box(frame)

        # Convert frame to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # Scale image to fit label while maintaining aspect ratio
        label_size = self.video_label.size()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label_size, 
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        
        # Center the pixmap in the label
        x = (label_size.width() - scaled_pixmap.width()) // 2
        y = (label_size.height() - scaled_pixmap.height()) // 2
        
        # Clear the label and set the new pixmap
        self.video_label.clear()
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setContentsMargins(x, y, x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.display_frame()

    def update_frame(self):
        if self.is_playing:
            self.read_frame()

    def previous_frame(self):
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 2))
        self.read_frame()

    def next_frame(self):
        self.read_frame()

    def confirm_selection(self):
        if self.box_drawer.box_drawn:
            self.target_box = self.box_drawer.get_box()
            self.start_tracking()
        else:
            self.statusBar().showMessage("Please draw a box first")

    def mousePressEvent(self, event):
        if self.video_label.underMouse():
            # Get the position relative to the video label
            pos = self.video_label.mapFrom(self, event.pos())
            
            # Get the actual video display area within the label
            label_size = self.video_label.size()
            pixmap_size = self.video_label.pixmap().size()
            
            # Calculate margins (if video is centered in label)
            x_margin = (label_size.width() - pixmap_size.width()) // 2
            y_margin = (label_size.height() - pixmap_size.height()) // 2
            
            # Adjust position by subtracting margins
            adjusted_x = pos.x() - x_margin
            adjusted_y = pos.y() - y_margin
            
            # Convert to original video coordinates
            scale_x = self.current_frame.shape[1] / pixmap_size.width()
            scale_y = self.current_frame.shape[0] / pixmap_size.height()
            
            # Calculate final coordinates
            x = max(0, min(int(adjusted_x * scale_x), self.current_frame.shape[1]))
            y = max(0, min(int(adjusted_y * scale_y), self.current_frame.shape[0]))
            
            self.box_drawer.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, None, None)

    def mouseMoveEvent(self, event):
        if self.video_label.underMouse() and self.box_drawer.drawing:
            # Get the position relative to the video label
            pos = self.video_label.mapFrom(self, event.pos())
            
            # Get the actual video display area within the label
            label_size = self.video_label.size()
            pixmap_size = self.video_label.pixmap().size()
            
            # Calculate margins (if video is centered in label)
            x_margin = (label_size.width() - pixmap_size.width()) // 2
            y_margin = (label_size.height() - pixmap_size.height()) // 2
            
            # Adjust position by subtracting margins
            adjusted_x = pos.x() - x_margin
            adjusted_y = pos.y() - y_margin
            
            # Convert to original video coordinates
            scale_x = self.current_frame.shape[1] / pixmap_size.width()
            scale_y = self.current_frame.shape[0] / pixmap_size.height()
            
            # Calculate final coordinates
            x = max(0, min(int(adjusted_x * scale_x), self.current_frame.shape[1]))
            y = max(0, min(int(adjusted_y * scale_y), self.current_frame.shape[0]))
            
            self.box_drawer.mouse_callback(cv2.EVENT_MOUSEMOVE, x, y, None, None)
            self.display_frame()

    def mouseReleaseEvent(self, event):
        if self.video_label.underMouse():
            # Get the position relative to the video label
            pos = self.video_label.mapFrom(self, event.pos())
            
            # Get the actual video display area within the label
            label_size = self.video_label.size()
            pixmap_size = self.video_label.pixmap().size()
            
            # Calculate margins (if video is centered in label)
            x_margin = (label_size.width() - pixmap_size.width()) // 2
            y_margin = (label_size.height() - pixmap_size.height()) // 2
            
            # Adjust position by subtracting margins
            adjusted_x = pos.x() - x_margin
            adjusted_y = pos.y() - y_margin
            
            # Convert to original video coordinates
            scale_x = self.current_frame.shape[1] / pixmap_size.width()
            scale_y = self.current_frame.shape[0] / pixmap_size.height()
            
            # Calculate final coordinates
            x = max(0, min(int(adjusted_x * scale_x), self.current_frame.shape[1]))
            y = max(0, min(int(adjusted_y * scale_y), self.current_frame.shape[0]))
            
            self.box_drawer.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, None, None)
            self.display_frame()

    def start_tracking(self):
        # Switch to tracking phase
        self.statusBar().showMessage("Tracking initialized. Processing video...")
        self.box_drawer = BoundingBoxDrawer()

        # Disable controls during tracking
        self.play_button.setEnabled(False)
        self.confirm_button.setEnabled(False)
        
        # Create and start tracking thread
        self.tracking_thread = TrackingThread(
            self.video_path,
            self.target_box,
            self.detector,
            self.tracker,
            self.output_path,
            self.cropped_output_path
        )
        
        # Connect signals
        self.tracking_thread.frame_ready.connect(self.update_tracking_display)
        self.tracking_thread.progress_update.connect(self.statusBar().showMessage)
        self.tracking_thread.tracking_complete.connect(self.tracking_finished)
        
        # Start tracking
        self.tracking_thread.start()

    def update_tracking_display(self, frame):
        # Update the video display with the processed frame
        self.current_frame = frame
        self.display_frame()

    def tracking_finished(self):
        self.statusBar().showMessage("Tracking completed!")
        self.play_button.setEnabled(True)
        
    def closeEvent(self, event):
        # Make sure to stop the tracking thread when closing the window
        if hasattr(self, 'tracking_thread'):
            self.tracking_thread.stop()
            self.tracking_thread.wait()
        super().closeEvent(event)

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

def main():
    MODEL_PATH = "models/YOLOv11.pt"
    VIDEO_PATH = "videos/parked-cars.mp4"
    OUTPUT_PATH = "output/tracked_video.mp4"
    CROPPED_OUTPUT_PATH = "output/cropped_number_plate.mp4"
    
    app = QApplication(sys.argv)
    window = VideoPlayerWindow(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH, CROPPED_OUTPUT_PATH)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()