import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from tracking import CentroidTracker
from roi import RegionAdjuster
from collections import OrderedDict
from typing import List

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

class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    def detect_motion(self, frame: np.ndarray) -> List[BoundingBox]:
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

class EdgeService:
    def __init__(self, car_model_path, plate_model_path, confidence_threshold=0.5):
        self.motion_detector = MotionDetector()
        self.car_model = YOLO(car_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.tracker = CentroidTracker()
        self.region_adjuster = RegionAdjuster(800, 600)
        self.confidence_threshold = confidence_threshold
        self.active = False

    def off(self):
        self.active = False

    def on(self, device='CPU'):
        self.active = True

    def predict(self, frame, CB):
        if not self.active:
            CB(None)
            return

        try:
            roi_frame = frame #self.region_adjuster.apply_roi_mask(frame)
            motion_boxes = self.motion_detector.detect_motion(roi_frame)
            detected_cars = self.detect_moving_cars(roi_frame, motion_boxes)
            detected_objects = self.detect_license_plate_boxes(roi_frame, detected_cars)

            CB(detected_objects)

        except Exception as e:
            print("Edge service error:", e)
            CB(OrderedDict())

    def detect_moving_cars(self, frame, motion_boxes):
        results = self.car_model(frame)[0].boxes
        detections = []

        for box in results:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detected_car_box = BoundingBox(x1, y1, x2 - x1, y2 - y1, confidence)
            if confidence > self.confidence_threshold and class_id in [2, 3, 5, 7]:
                for motion_box in motion_boxes:
                    if detected_car_box.intersects_with(motion_box):
                        detections.append((x1, y1, x2, y2))
                        break

        filtered_detections = self.tracker.non_max_suppression_fast(detections)
        tracked_cars = self.tracker.update(filtered_detections)
        return tracked_cars

    def detect_license_plate_boxes(self, frame, detected_cars):
        plates_results = self.plate_model(frame)[0].boxes

        for plate in plates_results:
            confidence = plate.conf[0].item()
            class_id = int(plate.cls[0].item())
            x1, y1, x2, y2 = map(int, plate.xyxy[0])

            if confidence > self.confidence_threshold and int(class_id) == 0:
                plate_box = BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1), confidence)
                best_match_car_id = self.match_plate_to_car(plate_box, detected_cars)

                if best_match_car_id is not None:
                    detected_cars[best_match_car_id]["plate_bbox"] = (int(x1), int(y1), int(x2), int(y2))

        return detected_cars

    def match_plate_to_car(self, plate_box, detected_cars):
        best_match_car_id = None
        smallest_distance = float('inf')

        for car_id, car_details in detected_cars.items():
            car_center = car_details["centroid"]
            car_box = car_details['bbox']

            plate_center_x = plate_box.x + plate_box.width / 2
            plate_center_y = plate_box.y + plate_box.height / 2

            distance = ((car_center[0] - plate_center_x) ** 2 + (car_center[1] - plate_center_y) ** 2) ** 0.5

            if self.is_spatially_aligned(car_box, plate_box) and distance < smallest_distance:
                smallest_distance = distance
                best_match_car_id = car_id

        return best_match_car_id

    def is_spatially_aligned(self, car_box, plate_box):
        car_height = car_box[3] - car_box[1]
        car_bottom_y = car_box[3]
        plate_bottom_y = plate_box.y + plate_box.height

        vertically_aligned = car_bottom_y - plate_bottom_y < car_height / 3
        horizontally_aligned = (
            plate_box.x > car_box[0] and
            plate_box.x + plate_box.width < car_box[2]
        )

        return vertically_aligned and horizontally_aligned
