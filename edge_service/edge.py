from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from tracking import CentroidTracker
from roi import RegionAdjuster
from typing import List
import threading
import utils
from bounding_box import BoundingBox

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
            if cv2.contourArea(contour) > 2500:  # Filter by area
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append(BoundingBox(x, y, w, h))

        self.prev_frame = gray
        
        merged_boxes = utils.merge_boxes(bounding_boxes)
        
        return merged_boxes

class EdgeService:
    def __init__(self, car_model_path, plate_model_path, confidence_threshold=0.2):
        self.motion_detector = MotionDetector()
        self.car_model = YOLO(car_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.tracker = CentroidTracker()
        self.region_adjuster = RegionAdjuster(800, 600)
        self.confidence_threshold = confidence_threshold
        self.active = False
        self.lock = threading.RLock()  

    def off(self):
        self.active = False

    def on(self, device='CPU'):
        self.active = True

    def predict(self, frame, CB):
        if not self.active:
            CB(dict())
            return

        try:
            roi_frame = frame #self.region_adjuster.apply_roi_mask(frame)
            motion_boxes = self.motion_detector.detect_motion(roi_frame)
            detected_cars = self.detect_moving_cars(roi_frame, motion_boxes)
            detected_plates = self.detect_license_plate_boxes(roi_frame, detected_cars)

            CB(detected_plates)

        except Exception as e:
            print("Edge prediction error:", e)
            CB(dict())

    def detect_moving_cars(self, frame, motion_boxes):

        if len(motion_boxes) == 0:
            tracked_cars = self.tracker.update([])
            return tracked_cars

        results = self.car_model(frame)[0].boxes
            
        detections = []

        for box in results:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detected_car_box = BoundingBox(x1, y1, x2 - x1, y2 - y1, confidence)
            if confidence > self.confidence_threshold and class_id in [2, 3, 5, 7]:
                for motion_box in motion_boxes:
                    if utils.intersect_over_union(detected_car_box, motion_box) >= 0.5:
                        detections.append((x1, y1, x2, y2))
                        break

        filtered_detections = self.tracker.non_max_suppression_fast(detections)
        tracked_cars = self.tracker.update(filtered_detections)

        return tracked_cars

    def detect_license_plate_boxes(self, frame, detected_cars):

        if len(detected_cars.items()) == 0:
            return {}

        plates_results = self.plate_model(frame)[0].boxes

        car_plates = {}
        for plate in plates_results:
            confidence = plate.conf[0].item()
            class_id = int(plate.cls[0].item())
            x1, y1, x2, y2 = map(int, plate.xyxy[0])

            if confidence > self.confidence_threshold and int(class_id) == 0:
                plate_box = BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1), confidence)
                best_match_car_id = self.match_plate_to_car(plate_box, detected_cars)

                if best_match_car_id is not None and not self.tracker.objects[best_match_car_id]["done"]:
                    car_plates[best_match_car_id] = (int(x1), int(y1), int(x2), int(y2))

        return car_plates

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
    
    def update_tracked_vehicle(self, vehicle_id, ocr_text, ocr_confidence):
        vehicle_details = self.tracker.objects[vehicle_id]
        prev_plate_number = vehicle_details["plate_number"]
        prev_confidence = vehicle_details["confidence"]
        occurs = vehicle_details["occurs"]

        if ocr_text and (ocr_confidence >= prev_confidence): 
            vehicle_details['plate_number'] = ocr_text
            vehicle_details["confidence"] = ocr_confidence
            vehicle_details["last_timestamp"] = datetime.now()
            if prev_plate_number == ocr_text:
                vehicle_details["occurs"] = occurs + 1
            else:
                vehicle_details["occurs"] = 0
            vehicle_details["done"] =  vehicle_details["occurs"] >= 2
            self.tracker.update_tracked_plate(vehicle_id, ocr_text)

    def log_results(self):
        print("LOGGING CURRENT RESULTS:")
        for object_id, plate in self.tracker.tracked_plates.items():
            print(f"ID: {object_id} - Plate: {plate}")

    def visualize(self, frame: np.ndarray, authorized: bool):
        original_h, original_w = frame.shape[:2]
        new_h = 600
        new_w = 800

        # Resize the frame first
        resized_frame = cv2.resize(frame, (new_w, new_h))

        # Compute scale factors
        scale_x = new_w / original_w
        scale_y = new_h / original_h

        for object_id, data in self.tracker.objects.items():
            centroid = data["centroid"]
            bbox = data["bbox"]
            plate_number = data["plate_number"]
            plate_confidence = data["confidence"]
            direction = data["direction"]

            # Scale coordinates
            x1, y1, x2, y2 = [int(coord * scale) for coord, scale in zip(bbox, [scale_x, scale_y, scale_x, scale_y])]
            cx, cy = int(centroid[0] * scale_x), int(centroid[1] * scale_y)

            box_color = (0, 255, 0) if authorized == 1 else ((0, 0, 255) if authorized == -1 else (0, 0, 255))

            # Draw on the resized frame
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), box_color, 1)
            cv2.circle(resized_frame, (cx, cy), 5, box_color, -1)
            text = f"ID {object_id} - {direction}"
            cv2.putText(resized_frame, text, (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            if plate_number:
                cv2.putText(resized_frame, f"Plate: {plate_number} - {plate_confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return resized_frame
