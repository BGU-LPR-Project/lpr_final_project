import threading
import queue
from roi import RegionAdjuster
from tracking import CentroidTracker
from ultralytics import YOLO
from datetime import datetime
import paddleocr
import cv2
from formats import process_plate
from util import resize_plate, sharpenHBF
import numpy as np

class BoundingBox:
    def __init__(self, x: int, y: int, width: int, height: int, confidence: float = 1.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence

    def intersects_with(self, other: 'BoundingBox') -> bool:
        return not (
            self.x + self.width < other.x or
            self.x > other.x + other.width or
            self.y + self.height < other.y or
            self.y > other.y + other.height
        )

    def merge_with(self, other: 'BoundingBox') -> 'BoundingBox':
        new_x = min(self.x, other.x)
        new_y = min(self.y, other.y)
        new_w = max(self.x + self.width, other.x + other.width) - new_x
        new_h = max(self.y + self.height, other.y + other.height) - new_y
        return BoundingBox(new_x, new_y, new_w, new_h, max(self.confidence, other.confidence))

class CarDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.0):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker = CentroidTracker()

    def detect_moving_cars(self, visualize_frame, frame, motion_boxes):
        moving_cars = []
        results = self.model(frame)
        detected_cars = results[0].boxes
        detections = []
        for box in detected_cars:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if confidence > self.confidence_threshold and class_id in [2,3,5,7]:
                car_box = BoundingBox(x1, y1, x2 - x1, y2 - y1, confidence)
                for motion_box in motion_boxes:
                    if car_box.intersects_with(motion_box):
                        moving_cars.append(car_box)
                        detections.append((x1,y1,x2,y2))
                        break
        filtered = self.tracker.non_max_suppression_fast(detections)
        return self.tracker.update(filtered)

class LicensePlateDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.reader = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')

    def detect_license_plates(self, visualize_frame, frame, detected_cars, is_in_entrance_or_exit):
        license_plates = self.model(frame)[0]
        for plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = plate
            if conf > self.confidence_threshold and int(cls_id) == 0:
                plate_box = BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1), conf)
                match_id = self.match_plate_to_car(plate_box, detected_cars)
                if match_id is not None:
                    car = detected_cars[match_id]
                    cropped = frame[int(y1):int(y2), int(x1):int(x2)]
                    text, text_conf = self.read_text_from_plate(cropped, match_id)
                    processed = process_plate(text) if text else None
                    if processed and text_conf > car["confidence"]:
                        car['plate_number'] = processed
                        car['confidence'] = text_conf
                        car['last_timestamp'] = datetime.now()
                        car['direction'] = is_in_entrance_or_exit((x1, y1, x2, y2))
        return detected_cars

    def match_plate_to_car(self, plate_box, detected_cars):
        best_id = None
        min_dist = float('inf')
        for car_id, data in detected_cars.items():
            car_box = data['bbox']
            car_center = data['centroid']
            px, py = plate_box.x + plate_box.width/2, plate_box.y + plate_box.height/2
            dist = ((car_center[0]-px)**2 + (car_center[1]-py)**2) ** 0.5
            if self.is_spatially_aligned(car_box, plate_box) and dist < min_dist:
                best_id = car_id
                min_dist = dist
        return best_id

    def is_spatially_aligned(self, car_box, plate_box):
        car_h = car_box[3] - car_box[1]
        bottom_diff = car_box[3] - (plate_box.y + plate_box.height)
        vertical = bottom_diff < car_h / 3
        horizontal = plate_box.x > car_box[0] and plate_box.x + plate_box.width < car_box[2]
        return vertical and horizontal

    def read_text_from_plate(self, cropped, car_id, threshold=0.8):
        img = resize_plate(cropped)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        sharp = sharpenHBF(blur)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        try:
            results = self.reader.ocr(sharp)
            high = [(line[1][0], line[1][1]) for line in results[0] if line[1][1] >= threshold]
            return max(high, key=lambda x: x[1]) if high else (None, 0.0)
        except:
            return None, 0.0

class EdgeService:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.thread = None
        self.active = False
        self.car_detector = CarDetector("models/yolo11n.pt")
        self.lp_detector = LicensePlateDetector("models/license_plate_detector.pt")
        self.region_adjuster = RegionAdjuster(800, 600)

    def on(self, device='CPU'):
        self.active = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def off(self):
        self.active = False
        if self.thread:
            self.thread.join()

    def predict(self, predictData, CB):
        frame, motion_boxes = predictData
        cars = self.car_detector.detect_moving_cars(frame, frame, motion_boxes)
        detections = self.lp_detector.detect_license_plates(frame, frame, cars, self.region_adjuster.is_in_entrance_or_exit)
        result = (frame, detections)
        if CB:
            CB(result)
        return result

    def _run(self):
        while self.active:
            try:
                item = self.input_queue.get(timeout=1)
                result = self.predict(item, None)
                self.output_queue.put(result)
                self.input_queue.task_done()
            except queue.Empty:
                continue