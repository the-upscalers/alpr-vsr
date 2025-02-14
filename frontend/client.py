# client/main.py
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QSizePolicy,
    QFileDialog,
    QProgressBar,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import sys
import cv2
import requests
from pathlib import Path
import json
from dotenv import load_dotenv
import os

load_dotenv()
SERVER_URL = os.getenv("SERVER_URL")


class VideoUploadThread(QThread):
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    progress_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    start_timer = pyqtSignal(int)

    def __init__(self, video_path, bbox, server_url, timer):
        super().__init__()
        self.video_path = video_path
        self.bbox = bbox
        self.server_url = server_url
        self.task_id = None
        self.timer = timer
        self.timer.timeout.connect(self.check_progress)

    def check_progress(self):
        if not self.task_id:
            return

        response = requests.get(f"{self.server_url}/task_status/{self.task_id}")
        data = response.json()
        progress = data.get("progress", 0)

        self.progress_update.emit(progress)
        self.status_update.emit(f"Progress: {progress}% ({data.get('status')})")

        print(data)

        if data.get("status") == "SUCCESS":
            self.timer.stop()
            self.status_update.emit("Processing Complete! Ready to download.")
            self.download_btn.setEnabled(True)

    def run(self):
        try:
            self.status_update.emit("Uploading video...")

            # Prepare the files and data for upload
            with open(self.video_path, "rb") as video_file:
                files = {"video": video_file}
                data = {"bbox": json.dumps(self.bbox)}

                # Make the request
                response = requests.post(
                    f"{self.server_url}/process-video", files=files, data=data
                )

            if response.status_code == 200:
                self.task_id = response.json()["task_id"]
                self.status_update.emit(
                    f"Video uploaded successfully! Task ID: {self.task_id}"
                )
                self.start_timer.emit(2000)  # Poll every 2 seconds
            else:
                self.error_occurred.emit(f"Server error: {response.text}")

        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")


class VideoDownloadThread(QThread):
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, task_id, server_url):
        super().__init__()
        self.task_id = task_id
        self.server_url = server_url

    def run(self):
        try:
            response = requests.get(f"{self.server_url}/download/{self.task_id}")
            if response.status_code == 200:
                output_path = Path.home() / "Downloads" / f"{self.task_id}.mp4"
                with open(output_path, "wb") as f:
                    f.write(response.content)
                self.status_update.emit("Output saved to: " + str(output_path))
            else:
                raise Exception(f"Server error: {response.text}")

        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")


class VideoPlayerWindow(QMainWindow):
    def __init__(self, server_url=SERVER_URL):
        super().__init__()
        self.server_url = server_url
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.timer = QTimer()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("License Plate Tracker - Client")
        self.setMinimumSize(1600, 900)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Add upload button
        self.upload_button = QPushButton("Upload Video")
        self.upload_button.setStyleSheet(
            """
            QPushButton {
                padding: 8px 16px;
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )
        self.upload_button.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_button)

        # Create video display container
        video_container = QWidget()
        video_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        # Create video display
        self.video_label = QLabel()
        self.video_label.setFrameStyle(QFrame.Shape.Box)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("QLabel { background-color: #f0f0f0; }")
        video_layout.addWidget(self.video_label)
        layout.addWidget(video_container)

        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """
        )
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Create controls widget
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)

        # Create control buttons
        button_style = """
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
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """

        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.setCheckable(True)
        self.play_button.setStyleSheet(button_style)

        self.prev_frame_button = QPushButton("◀")
        self.prev_frame_button.setFixedWidth(50)

        self.next_frame_button = QPushButton("▶")
        self.next_frame_button.setFixedWidth(50)

        self.confirm_button = QPushButton("Process Video")
        self.confirm_button.setEnabled(False)
        self.confirm_button.setStyleSheet(button_style)

        self.download_button = QPushButton("Download Output")
        self.download_button.setEnabled(False)
        self.download_button.setStyleSheet(button_style)

        # Add buttons to controls layout
        controls_layout.addStretch()
        controls_layout.addWidget(self.prev_frame_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.next_frame_button)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.confirm_button)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.download_button)
        controls_layout.addStretch()

        # Add controls to main layout
        layout.addWidget(controls_widget)

        # Add status bar
        self.statusBar().showMessage("Upload a video to begin")

        # Connect button signals
        self.play_button.clicked.connect(self.toggle_play)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.confirm_button.clicked.connect(self.process_video)
        self.download_button.clicked.connect(self.download_output)

        # Setup video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False

        # Initialize drawing state
        self.drawing = False
        self.bbox = None
        self.start_point = None

    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )

        if file_name:
            if self.cap is not None:
                self.cap.release()

            self.video_path = file_name
            self.cap = cv2.VideoCapture(self.video_path)

            if self.cap.isOpened():
                self.play_button.setEnabled(True)
                self.bbox = None
                self.confirm_button.setEnabled(False)
                self.read_frame()
                self.statusBar().showMessage("Draw a box around the license plate")
            else:
                QMessageBox.critical(self, "Error", "Could not open video file")

    def read_frame(self):
        if self.cap is None:
            return

        ret, self.current_frame = self.cap.read()
        if ret:
            self.display_frame()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, self.current_frame = self.cap.read()
            if not ret:
                self.play_button.setChecked(False)
                self.is_playing = False
                self.timer.stop()

    def display_frame(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()

        # Draw current box if exists
        if self.bbox is not None:
            cv2.rectangle(
                frame,
                (self.bbox[0], self.bbox[1]),
                (self.bbox[2], self.bbox[3]),
                (0, 255, 0),
                2,
            )

        # Convert frame to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        ).rgbSwapped()

        # Scale image to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Center the pixmap in the label
        x = (self.video_label.width() - scaled_pixmap.width()) // 2
        y = (self.video_label.height() - scaled_pixmap.height()) // 2

        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setContentsMargins(x, y, x, y)

    def toggle_play(self):
        self.is_playing = self.play_button.isChecked()
        if self.is_playing:
            self.timer.start(int(1000 / 30))  # 30 fps
            self.play_button.setText("Pause")
            self.statusBar().showMessage("Playing video")
        else:
            self.timer.stop()
            self.play_button.setText("Play")
            self.statusBar().showMessage("Paused - Draw a box around the license plate")

    def update_frame(self):
        if self.is_playing:
            self.read_frame()

    def previous_frame(self):
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 2))
        self.read_frame()

    def next_frame(self):
        self.read_frame()

    def mousePressEvent(self, event):
        if (
            self.video_label.underMouse()
            and not self.is_playing
            and self.current_frame is not None
        ):
            pos = self.video_label.mapFrom(self, event.pos())
            self.drawing = True
            self.start_point = self.convert_coordinates(pos)
            self.bbox = None
            self.confirm_button.setEnabled(False)

    def mouseMoveEvent(self, event):
        if self.drawing and self.video_label.underMouse():
            pos = self.video_label.mapFrom(self, event.pos())
            current_point = self.convert_coordinates(pos)
            self.bbox = [
                min(self.start_point[0], current_point[0]),
                min(self.start_point[1], current_point[1]),
                max(self.start_point[0], current_point[0]),
                max(self.start_point[1], current_point[1]),
            ]
            self.display_frame()

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            if self.bbox is not None:
                self.confirm_button.setEnabled(True)
                self.statusBar().showMessage(
                    "Box drawn - Click 'Process Video' to begin tracking"
                )

    def convert_coordinates(self, pos):
        """Convert Qt coordinates to video coordinates"""
        label_size = self.video_label.size()
        pixmap_size = self.video_label.pixmap().size()

        # Calculate margins
        x_margin = (label_size.width() - pixmap_size.width()) // 2
        y_margin = (label_size.height() - pixmap_size.height()) // 2

        # Adjust position
        adjusted_x = pos.x() - x_margin
        adjusted_y = pos.y() - y_margin

        # Convert to original video coordinates
        scale_x = self.current_frame.shape[1] / pixmap_size.width()
        scale_y = self.current_frame.shape[0] / pixmap_size.height()

        x = max(0, min(int(adjusted_x * scale_x), self.current_frame.shape[1]))
        y = max(0, min(int(adjusted_y * scale_y), self.current_frame.shape[0]))

        return (x, y)

    def start_timer(self, interval):
        self.timer.start(interval)

    def process_video(self):
        if self.video_path and self.bbox:
            self.progress_bar.setVisible(True)
            self.confirm_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.upload_button.setEnabled(False)

            # Create and start upload thread
            self.upload_thread = VideoUploadThread(
                self.video_path,
                {
                    "x1": self.bbox[0],
                    "y1": self.bbox[1],
                    "x2": self.bbox[2],
                    "y2": self.bbox[3],
                },
                self.server_url,
                self.timer,
            )
            self.upload_thread.status_update.connect(self.update_status)
            self.upload_thread.progress_update.connect(self.update_progress)
            self.upload_thread.progress_complete.connect(self.processing_complete)
            self.upload_thread.error_occurred.connect(self.handle_error)
            self.upload_thread.start_timer.connect(self.start_timer)
            self.upload_thread.start()

    def download_output(self):
        if self.upload_thread and self.upload_thread.task_id:
            self.download_button.setEnabled(False)
            self.download_thread = VideoDownloadThread(
                self.upload_thread.task_id, self.server_url
            )
            self.download_thread.status_update.connect(self.update_status)
            self.download_thread.error_occurred.connect(self.handle_error)
            self.download_thread.start()

    def update_status(self, message):
        self.statusBar().showMessage(message)
        self.progress_bar.setFormat(message)

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def processing_complete(self, output_path):
        self.progress_bar.setVisible(False)
        self.play_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.download_button.setEnabled(True)
        self.statusBar().showMessage(
            f"Processing complete! Click 'Download Output' to save the result"
        )

    def handle_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.play_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.confirm_button.setEnabled(True)
        self.statusBar().showMessage(f"Error: {error_message}")

        QMessageBox.critical(
            self, "Error", f"An error occurred during processing:\n{error_message}"
        )

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = VideoPlayerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
