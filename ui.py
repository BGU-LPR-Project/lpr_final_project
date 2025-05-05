import sys
import time
import pickle
import redis
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class RedisFrameViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge Visual Output")
        self.label = QLabel(self)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)

        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.queue_name = "visual_frame_queue"

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # 100 ms = 10 FPS

    def update_frame(self):
        print(self.redis.llen("visual_frame_queue"))
        frame_data = self.redis.lpop(self.queue_name)
        if frame_data:
            try:
                frame = pickle.loads(frame_data)
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.label.setPixmap(QPixmap.fromImage(qimg))
            except Exception as e:
                print(f"Failed to load or display frame: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = RedisFrameViewer()
    viewer.show()
    sys.exit(app.exec_())
