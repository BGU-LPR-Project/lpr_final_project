# import sys
# import time
# import pickle
# import redis
# import cv2
# import numpy as np
# import requests
# from PyQt5.QtWidgets import (
#     QApplication, QLabel, QWidget, QVBoxLayout,
#     QHBoxLayout, QPushButton
# )
# from PyQt5.QtCore import QTimer
# from PyQt5.QtGui import QImage, QPixmap


# class RedisFrameViewer(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Edge Visual Output")

#         self.redis = redis.Redis(host='localhost', port=6379, db=0)
#         self.queue_name = "visual_frame_queue"
#         self.video_path = "/app/recordings/fullMovie20250420_09.mp4"

#         # UI Label
#         self.label = QLabel(self)
#         self.label.setFixedSize(640, 480)

#         # Buttons
#         self.start_button = QPushButton("Start")
#         self.pause_button = QPushButton("Pause")
#         self.resume_button = QPushButton("Resume")
#         self.stop_button = QPushButton("Stop")
#         self.restart_button = QPushButton("Restart")
#         self.skip_button = QPushButton("Skip 10s")  # NEW BUTTON

#         # Connect buttons to handlers
#         self.start_button.clicked.connect(self.handle_start)
#         self.pause_button.clicked.connect(self.handle_pause)
#         self.resume_button.clicked.connect(self.handle_resume)
#         self.stop_button.clicked.connect(self.handle_stop)
#         self.restart_button.clicked.connect(self.handle_restart)
#         self.skip_button.clicked.connect(self.handle_skip)  # NEW HANDLER

#         # Layout setup
#         btn_layout = QHBoxLayout()
#         for btn in [
#             self.start_button, self.pause_button,
#             self.resume_button, self.stop_button,
#             self.restart_button, self.skip_button  # INCLUDE SKIP BUTTON
#         ]:
#             btn_layout.addWidget(btn)

#         layout = QVBoxLayout()
#         layout.addWidget(self.label)
#         layout.addLayout(btn_layout)
#         self.setLayout(layout)

#         # Timer for frame updates
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(100)  # 10 FPS (every 100 ms)

#         # Auto-start
#         self.handle_start()

#     def call_api(self, endpoint):
#         url = f"http://localhost:8000{endpoint}"
#         try:
#             if endpoint == "/start-video":
#                 response = requests.post(url, json={"path": self.video_path}, timeout=5)
#             else:
#                 response = requests.post(url, timeout=5)
#             print(f"{endpoint} -> {response.status_code} - {response.text}")
#         except requests.RequestException as e:
#             print(f"Failed API call to {endpoint}: {e}")

#     def handle_start(self):
#         self.call_api("/start-video")

#     def handle_pause(self):
#         self.call_api("/pause-video")

#     def handle_resume(self):
#         self.call_api("/resume-video")

#     def handle_stop(self):
#         self.call_api("/stop-video")

#     def handle_restart(self):
#         self.call_api("/restart-video")

#     def handle_skip(self):
#         self.call_api("/skip-10s")  # NEW ENDPOINT CALL

#     def update_frame(self):
#         frame_data = self.redis.lpop(self.queue_name)
#         if frame_data:
#             try:
#                 frame = pickle.loads(frame_data)
#                 if frame is not None:
#                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     h, w, ch = frame_rgb.shape
#                     bytes_per_line = ch * w
#                     qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
#                     self.label.setPixmap(QPixmap.fromImage(qimg))
#             except Exception as e:
#                 print(f"Failed to load or display frame: {e}")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     viewer = RedisFrameViewer()
#     viewer.show()
#     sys.exit(app.exec_())
