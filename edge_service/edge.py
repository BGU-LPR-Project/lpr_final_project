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
import redis
import pickle


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
    def __init__(self, car_model_path, plate_model_path, car_conf_threshold=0.5, plate_conf_threshold=0.2):
        self.motion_detector = MotionDetector()
        self.car_model = YOLO(car_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.tracker = CentroidTracker()
        self.region_adjuster = RegionAdjuster(800, 600)
        
        self.car_conf_threshold = car_conf_threshold
        self.plate_conf_threshold = plate_conf_threshold

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
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            detected_car_box = BoundingBox(x1, y1, x2 - x1, y2 - y1, confidence)
            if confidence > self.car_conf_threshold and class_id in [2, 3, 5, 7]:
                for motion_box in motion_boxes:
                    if (
                        detected_car_box.intersects_with(motion_box) and
                        utils.motion_box_valid_for_car(detected_car_box, motion_box)
                    ):
                        detections.append((x1, y1, x2, y2))
                        break

        filtered_detections = self.tracker.non_max_suppression_fast(detections)
        tracked_cars = self.tracker.update(filtered_detections)

        return tracked_cars
    
    def detect_license_plate_boxes(self, frame, detected_cars):

        if len(detected_cars.items()) == 0:
            return {}

        plates_results = self.plate_model(frame)[0].boxes

        assigned_car_ids = []
        car_plates = {}
        for plate in plates_results:
            confidence = plate.conf[0].item()
            class_id = int(plate.cls[0].item())
            x1, y1, x2, y2 = map(int, plate.xyxy[0])
                        
            if confidence > self.plate_conf_threshold and int(class_id) == 0:
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                plate_box = BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1), confidence)
                best_match_car_id = self.match_plate_to_car(plate_box, detected_cars, assigned_car_ids)
                if best_match_car_id is not None:
                    assigned_car_ids.append(best_match_car_id)

                if best_match_car_id is not None and not self.tracker.objects[best_match_car_id]["done"]:
                    car_plates[best_match_car_id] = (int(x1), int(y1), int(x2), int(y2))

        return car_plates
    
    def match_plate_to_car(self, plate_box, detected_cars, assigned_car_ids):
        best_match_car_id = None
        smallest_distance = float('inf')

        plate_center_x = plate_box.x + plate_box.width / 2
        plate_center_y = plate_box.y + plate_box.height / 2

        for car_id, car_details in detected_cars.items():
            if car_id in assigned_car_ids:
                continue  # Skip already assigned cars

            car_center_x, car_center_y = car_details["centroid"]
            car_box = car_details['bbox']

            # Use the alignment logic
            if self.is_spatially_aligned(car_box, plate_box):
                distance = ((car_center_x - plate_center_x) ** 2 + (car_center_y - plate_center_y) ** 2) ** 0.5

                if distance < smallest_distance:
                    smallest_distance = distance
                    best_match_car_id = car_id

        return best_match_car_id
    
    def is_spatially_aligned(self, car_box, plate_box):
        car_x1, car_y1, car_x2, car_y2 = car_box
        plate_x1, plate_y1 = plate_box.x, plate_box.y
        plate_x2, plate_y2 = plate_box.x + plate_box.width, plate_box.y + plate_box.height

        # Allow plate to be anywhere vertically within car height
        vertically_aligned = plate_y1 >= car_y1 and plate_y2 <= car_y2

        # Allow plates closer to edges, add tolerance of few pixels outside the car box horizontally
        horizontal_tolerance = 0  # adjust this if needed
        horizontally_aligned = (
                plate_x2 >= car_x1 - horizontal_tolerance and
                plate_x1 <= car_x2 + horizontal_tolerance
        )

        return vertically_aligned and horizontally_aligned

    def update_tracked_vehicle(self, vehicle_id, ocr_text, ocr_confidence):
        vehicle = self.tracker.objects[vehicle_id]
        prev_text = vehicle["plate_number"]
        prev_conf = vehicle["confidence"]
        occurs = vehicle["occurs"]

        if ocr_text and ocr_confidence >= prev_conf:
            is_same_text = (ocr_text == prev_text)

            vehicle.update({
                "plate_number": ocr_text,
                "confidence": ocr_confidence,
                "last_timestamp": datetime.now(),
                "occurs": occurs + 1 if is_same_text else 0,
            })
            vehicle["done"] = vehicle["occurs"] >= 2
            self.tracker.update_tracked_plate(vehicle_id, ocr_text)
    
    def log_results(self):
        redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
        redis_client.set("tracked_plates", pickle.dumps([(oid, v["plate_number"], v["confidence"]) for oid, v in self.tracker.objects.items()]))

        print("LOGGING CURRENT RESULTS:")
        redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
        redis_client.set("tracked_plates", pickle.dumps([
            (object_id, v["plate_number"], v["confidence"])
            for object_id, v in self.tracker.objects.items()
            if v["plate_number"]
        ]))
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
