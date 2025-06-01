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
    """
    MotionDetector uses optical flow to detect motion between consecutive video frames.

    Attributes:
        prev_frame (np.ndarray or None): Grayscale image of the previous frame used for motion comparison.
    """

    def __init__(self):
        """
        Initialize MotionDetector with no previous frame.
        """
        self.prev_frame = None

    def detect_motion(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detects regions of motion in the given video frame using optical flow.

        Args:
            frame (np.ndarray): The current video frame in BGR color space.

        Returns:
            List[BoundingBox]: A list of bounding boxes around detected motion areas.
        """
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If no previous frame, store current and return no motion
        if self.prev_frame is None:
            self.prev_frame = gray
            return []

        # Calculate dense optical flow between previous and current grayscale frames
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )

        # Compute magnitude of flow vectors
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold magnitude to create a binary motion mask
        motion_mask = cv2.threshold(mag, 2, 255, cv2.THRESH_BINARY)[1]

        # Find contours from the binary motion mask
        contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        # Filter contours by area and create bounding boxes
        for contour in contours:
            if cv2.contourArea(contour) > 2500:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append(BoundingBox(x, y, w, h))

        # Update previous frame for next iteration
        self.prev_frame = gray

        # Merge overlapping or close bounding boxes using utility function
        merged_boxes = utils.merge_boxes(bounding_boxes)

        return merged_boxes

class EdgeService:
    """
    Service for detecting moving cars and license plates in video frames,
    tracking cars over time using motion detection and YOLO models.
    """

    def __init__(self, car_model_path, plate_model_path, car_conf_threshold=0.5, plate_conf_threshold=0.2):
        self.motion_detector = MotionDetector()  # Initialize motion detector
        self.car_model = YOLO(car_model_path)    # Load YOLO model for car detection
        self.plate_model = YOLO(plate_model_path)  # Load YOLO model for license plate detection
        self.tracker = CentroidTracker()         # Initialize centroid-based object tracker
        self.region_adjuster = RegionAdjuster(800, 600)  # Region of interest adjuster (currently unused)
        
        self.car_conf_threshold = car_conf_threshold    # Minimum confidence for car detections
        self.plate_conf_threshold = plate_conf_threshold  # Minimum confidence for plate detections

        self.active = False            # Service state flag
        self.lock = threading.RLock()  # Thread-safe lock for concurrent access

    def off(self):
        """Deactivate the service."""
        self.active = False

    def on(self, device='CPU'):
        """Activate the service (device param unused)."""
        self.active = True

    def predict(self, frame, CB):
        """
        Perform motion detection, car detection, and license plate detection on the input frame.
        Calls callback CB with detected license plates or empty dict if inactive or on error.
        """
        if not self.active:
            CB(dict())  # Return empty dict if service is inactive
            return

        try:
            roi_frame = frame  # Placeholder for potential ROI masking
            motion_boxes = self.motion_detector.detect_motion(roi_frame)  # Detect motion regions
            detected_cars = self.detect_moving_cars(roi_frame, motion_boxes)  # Detect cars within motion
            detected_plates = self.detect_license_plate_boxes(roi_frame, detected_cars)  # Detect plates for cars

            CB(detected_plates)  # Pass detected plates to callback

        except Exception as e:
            print("Edge prediction error:", e)
            CB(dict())  # Return empty dict on error

    def detect_moving_cars(self, frame, motion_boxes):
        """
        Detect cars in the frame overlapping with motion areas.
        Filters detections by confidence and class.
        Updates the tracker and returns currently tracked cars.
        """
        if len(motion_boxes) == 0:
            # No motion detected, update tracker with empty list
            return self.tracker.update([])

        results = self.car_model(frame)[0].boxes  # Run car detection model
        detections = []

        for box in results:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detected_car_box = BoundingBox(x1, y1, x2 - x1, y2 - y1, confidence)

            # Consider only confident detections of relevant vehicle classes
            if confidence > self.car_conf_threshold and class_id in [2, 3, 5, 7]:
                # Check if car detection intersects with any motion box and is valid
                for motion_box in motion_boxes:
                    if (
                        detected_car_box.intersects_with(motion_box) and
                        utils.motion_box_valid_for_car(detected_car_box, motion_box)
                    ):
                        detections.append((x1, y1, x2, y2))
                        break

        # Apply Non-Max Suppression to filter overlapping detections
        filtered_detections = self.tracker.non_max_suppression_fast(detections)
        # Update tracker with filtered car detections
        tracked_cars = self.tracker.update(filtered_detections)

        return tracked_cars
    
    def detect_license_plate_boxes(self, frame, detected_cars):
        """
        Detect license plates in the frame and assign them to tracked cars.
        Returns a dictionary mapping car IDs to license plate bounding boxes.
        """
        if len(detected_cars.items()) == 0:
            return {}  # No cars detected, return empty dict

        plates_results = self.plate_model(frame)[0].boxes  # Run plate detection model
        assigned_car_ids = []  # Track car IDs already assigned a plate
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

                # Only add plate if matched car is tracked and not marked done
                if best_match_car_id is not None and not self.tracker.objects[best_match_car_id]["done"]:
                    car_plates[best_match_car_id] = (int(x1), int(y1), int(x2), int(y2))

        return car_plates
    
    def match_plate_to_car(self, plate_box, detected_cars, assigned_car_ids):
        """
        Match a detected license plate box to the closest spatially aligned car.
        Skips cars already assigned a plate. Returns best matched car ID or None.
        """
        best_match_car_id = None
        smallest_distance = float('inf')

        plate_center_x = plate_box.x + plate_box.width / 2
        plate_center_y = plate_box.y + plate_box.height / 2

        for car_id, car_details in detected_cars.items():
            if car_id in assigned_car_ids:
                continue  # Skip cars that already have an assigned plate

            car_center_x, car_center_y = car_details["centroid"]
            car_box = car_details['bbox']

            # Check if plate is spatially aligned with car bounding box
            if self.is_spatially_aligned(car_box, plate_box):
                # Calculate Euclidean distance between plate and car centers
                distance = ((car_center_x - plate_center_x) ** 2 + (car_center_y - plate_center_y) ** 2) ** 0.5

                # Keep track of car with smallest distance to plate center
                if distance < smallest_distance:
                    smallest_distance = distance
                    best_match_car_id = car_id

        return best_match_car_id

    def is_spatially_aligned(self, car_box, plate_box):
        """
        Check if license plate box is vertically within and horizontally overlapping (with tolerance)
        the car bounding box.
        """
        car_x1, car_y1, car_x2, car_y2 = car_box
        plate_x1, plate_y1 = plate_box.x, plate_box.y
        plate_x2, plate_y2 = plate_box.x + plate_box.width, plate_box.y + plate_box.height

        # Plate must be fully within car vertically
        vertically_aligned = plate_y1 >= car_y1 and plate_y2 <= car_y2

        # Plate horizontally overlaps car box with zero tolerance (can be adjusted)
        horizontal_tolerance = 0
        horizontally_aligned = (
            plate_x2 >= car_x1 - horizontal_tolerance and
            plate_x1 <= car_x2 + horizontal_tolerance
        )

        return vertically_aligned and horizontally_aligned

    def update_tracked_vehicle(self, vehicle_id, ocr_text, ocr_confidence):
        """
        Update tracked vehicle info with new OCR plate text and confidence.
        Increments occurrence count if plate text matches previous; marks done if stable.
        """
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
        """
        Log current tracked plates to Redis and print them to console.
        Stores only plates with non-empty plate numbers.
        """
        redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
        redis_client.set("tracked_plates", pickle.dumps([
            (object_id, v["plate_number"], v["confidence"])
            for object_id, v in self.tracker.objects.items()
            if v["plate_number"]
        ]))
        print("LOGGING CURRENT RESULTS:")
        for object_id, plate in self.tracker.tracked_plates.items():
            print(f"ID: {object_id} - Plate: {plate}")

    def visualize(self, frame: np.ndarray, authorized: bool):
        """
        Resize frame to 800x600 and draw bounding boxes, centroids, directions,
        and plate info on tracked vehicles. Color boxes green if authorized,
        red otherwise.
        """
        original_h, original_w = frame.shape[:2]
        new_h = 600
        new_w = 800

        # Resize frame for visualization
        resized_frame = cv2.resize(frame, (new_w, new_h))

        # Compute scaling factors for coordinates
        scale_x = new_w / original_w
        scale_y = new_h / original_h

        for object_id, data in self.tracker.objects.items():
            centroid = data["centroid"]
            bbox = data["bbox"]
            plate_number = data["plate_number"]
            plate_confidence = data["confidence"]
            direction = data["direction"]

            # Scale bounding box and centroid coordinates
            x1, y1, x2, y2 = [int(coord * scale) for coord, scale in zip(bbox, [scale_x, scale_y, scale_x, scale_y])]
            cx, cy = int(centroid[0] * scale_x), int(centroid[1] * scale_y)

            # Choose box color based on authorization status
            box_color = (0, 255, 0) if authorized == 1 else ((0, 0, 255) if authorized == -1 else (0, 0, 255))

            # Draw bounding box and centroid
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), box_color, 1)
            cv2.circle(resized_frame, (cx, cy), 5, box_color, -1)

            # Draw ID and movement direction text
            text = f"ID {object_id} - {direction}"
            cv2.putText(resized_frame, text, (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            # Draw plate number and confidence if available
            if plate_number:
                cv2.putText(resized_frame, f"Plate: {plate_number} - {plate_confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return resized_frame
