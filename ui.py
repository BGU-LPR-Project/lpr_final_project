import sys
import time
import pickle
import redis
import cv2
import numpy as np
import requests
import os
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QFileDialog
)

ICON_DIR = os.path.join(os.path.dirname(__file__), "icons")

class RedisFrameViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge Visual Output")

        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.queue_name = "visual_frame_queue"
        self.video_path = "/app/recordings/fullMovie20250420_09.mp4"

        # Main video display label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1280, 720)
        # self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setScaledContents(True)

        self.init_ui()

    def closeEvent(self, event):
        self.call_api("/stop-video")
        print("UI closed — stop-video API called.")
        event.accept()  # Accept the close event so the window will close

    def init_ui(self):
        self.setWindowTitle("License Plate Recognition - LPR UI")
        self.setWindowIcon(QIcon(os.path.join(ICON_DIR, "lpr_icon.png")))

        # Buttons
        self.upload_button = QPushButton()
        self.start_button = QPushButton()
        self.pause_button = QPushButton()
        self.resume_button = QPushButton()
        self.restart_button = QPushButton()
        self.skip_button = QPushButton()

        icon_size = QSize(48, 48)
        self.upload_button.setToolTip("Load video from your computer")
        self.upload_button.setIcon(QIcon(os.path.join(ICON_DIR, "upload.png")))
        self.upload_button.setIconSize(icon_size)

        self.start_button.setToolTip("Start video playback")
        self.start_button.setIcon(QIcon(os.path.join(ICON_DIR, "play.png")))
        self.start_button.setIconSize(icon_size)

        self.pause_button.setToolTip("Pause video")
        self.pause_button.setIcon(QIcon(os.path.join(ICON_DIR, "pause.png")))
        self.pause_button.setIconSize(icon_size)

        self.resume_button.setToolTip("Resume video from current frame")
        self.resume_button.setIcon(QIcon(os.path.join(ICON_DIR, "replay.png")))
        self.resume_button.setIconSize(icon_size)

        self.restart_button.setToolTip("Restart video from the beginning")
        self.restart_button.setIcon(QIcon(os.path.join(ICON_DIR, "restart.png")))
        self.restart_button.setIconSize(icon_size)

        self.skip_button.setToolTip("Jump forward 10 seconds in the video")
        self.skip_button.setIcon(QIcon(os.path.join(ICON_DIR, "forward.png")))
        self.skip_button.setIconSize(icon_size)

        # Connect buttons to handlers
        self.start_button.clicked.connect(self.handle_start)
        self.pause_button.clicked.connect(self.handle_pause)
        self.resume_button.clicked.connect(self.handle_resume)
        self.restart_button.clicked.connect(self.handle_restart)
        self.skip_button.clicked.connect(self.handle_skip)
        self.upload_button.clicked.connect(self.handle_load_video)

        # Layouts
        btn_layout = QHBoxLayout()
        for btn in [
            self.upload_button, self.start_button, self.pause_button,
            self.resume_button, self.restart_button, self.skip_button
        ]:
            btn_layout.addWidget(btn)

        final_layout = QVBoxLayout()
        final_layout.addWidget(self.video_label)
        final_layout.addLayout(btn_layout)

        self.status_label = QLabel("Ready 🔍")
        self.status_label.setStyleSheet("font-size: 14px; color: #5E5E5E; padding: 8px; font-weight: bold;")
        final_layout.addWidget(self.status_label)

        self.setLayout(final_layout)

        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # 10 FPS

        # self.handle_start()

    def call_api(self, endpoint):
        url = f"http://localhost:8000{endpoint}"
        try:
            if endpoint == "/start-video":
                response = requests.post(url, json={"path": self.video_path}, timeout=5)
            else:
                response = requests.post(url, timeout=5)
            print(f"{endpoint} -> {response.status_code} - {response.text}")
        except requests.RequestException as e:
            print(f"Failed API call to {endpoint}: {e}")

    def handle_start(self):
        if not self.video_path:
            self.status_label.setText("Please upload a video first ❗")
            return
        self.call_api("/start-video")
        self.status_label.setText("Playing video ▶️")

    def handle_pause(self):
        self.call_api("/pause-video")
        self.status_label.setText("Video paused ⏸️")

    def handle_resume(self):
        self.call_api("/resume-video")
        self.status_label.setText("Video resumed ⏵")

    def handle_restart(self):
        self.call_api("/restart-video")
        self.status_label.setText("Video restarted 🔄")

    def handle_skip(self):
        self.call_api("/skip-10s")
        self.status_label.setText("Skipped ahead 10 seconds ⏩")
        
    def update_frame(self):
        frame_data = self.redis.lpop(self.queue_name)
        if frame_data:
            try:
                frame = pickle.loads(frame_data)
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(qimg))
            except Exception as e:
                print(f"Failed to load or display frame: {e}")

    def handle_load_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_path = file_path
            self.status_label.setText("Video loaded successfully ✔️")
        else:
            self.status_label.setText("No video file selected ❌")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Style
    light_style = """
    QWidget {
        background-color: #eaf4fb;
        color: #003B73;
        font-family: 'Segoe UI';
        font-size: 10pt;
    }

    QPushButton {
        background-color: #d4eafd;
        color: #003B73;
        font-weight: 600;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #a8cdee;
    }

    QPushButton:hover {
        background-color: #e4f4ff;
    }

    QPushButton:pressed {
        background-color: #bdddf5;
    }

    QLabel {
        font-size: 12pt;
        font-weight: 600;
        color: #003B73;
    }

    QToolTip {
        background-color: #f2fbff;
        color: #003B73;
        border: 1px solid #a0cbe8;
        padding: 5px;
        border-radius: 6px;
        font-size: 9pt;
    }
    """
    app.setStyleSheet(light_style)

    viewer = RedisFrameViewer()
    viewer.show()
    sys.exit(app.exec_())
