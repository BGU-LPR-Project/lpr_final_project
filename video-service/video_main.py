import cv2
import zmq
import base64
import pickle
import numpy as np
import json
import os

class BoundingBox:
    def __init__(self, x: int, y: int, width: int, height: int, confidence: float = 1.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence

    def to_coords(self):
        return self.x, self.y, self.x + self.width, self.y + self.height

class MotionDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)
        self.prev_frame = None

    def detect_motion(self, frame, roi_masks):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray
            return []

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = cv2.threshold(mag, 2, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        if roi_masks is not None and motion_mask.shape != roi_masks.shape:
            roi_masks = cv2.resize(roi_masks, motion_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)

        if roi_masks is not None:
            motion_mask = cv2.bitwise_and(motion_mask, roi_masks)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append(BoundingBox(x, y, w, h))

        self.prev_frame = gray
        return boxes


def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def mask_from_rois(rois, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for mode in ["entrance", "exit"]:
        for x1, y1, x2, y2 in rois.get(mode, []):
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def load_rois(config_file="roi_config.json"):
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    return None

def main(video_path="/recordings/motion4.mp4", config_file="roi_config.json", check_interval=5):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://edge:5555")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame from video.")

    rois = load_rois(config_file)
    if rois is None:
        raise RuntimeError("ROI config file is missing. Please create 'roi_config.json' manually.")

    roi_mask = mask_from_rois(rois, first_frame.shape[:2])
    motion_detector = MotionDetector()

    print("[VIDEO] Starting motion detection...")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % check_interval != 0:
            continue

        motion_boxes = motion_detector.detect_motion(frame.copy(), roi_mask)

        if len(motion_boxes) > 0:
            encoded = encode_frame(frame)
            socket.send(pickle.dumps({"frame": encoded}))
            print(f"[VIDEO] Frame sent with {len(motion_boxes)} motion boxes")

    cap.release()
    print("[VIDEO] Finished.")

if __name__ == "__main__":
    main("/recordings/motion4.mp4", check_interval=5)
