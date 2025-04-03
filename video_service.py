import cv2
import threading
import queue
from edge_service import BoundingBox
import numpy as np

class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    def detect_motion(self, visualize_frame, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return []

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = cv2.threshold(mag, 2, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append(BoundingBox(x, y, w, h))

        self.prev_frame = gray
        return bounding_boxes

class VideoService:
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture = None
        self.motion_detector = MotionDetector()
        self.queue = queue.Queue()
        self.thread = None
        self.active = False

    def on(self, device='CPU'):
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            raise RuntimeError("Failed to open video")
        self.active = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def off(self):
        self.active = False
        if self.thread:
            self.thread.join()
        if self.capture:
            self.capture.release()

    def predict(self, predictData, CB):
        try:
            item = self.queue.get(timeout=1)
            if CB:
                CB(item)
            return item
        except queue.Empty:
            if CB:
                CB(None)
            return None

    def _run(self):
        while self.active:
            ret, frame = self.capture.read()
            if not ret:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            motion_boxes = self.motion_detector.detect_motion(frame, frame)
            if motion_boxes:
                self.queue.put((frame, motion_boxes))
