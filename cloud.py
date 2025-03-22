import cv2
import numpy as np
from typing import List
from ultralytics import YOLO
from edge import BoundingBox
import paddleocr
from tracking import CentroidTracker
from collections import OrderedDict
import util
from formats import *
from datetime import datetime
from typing import Dict


class CarDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.0):
        """
        Initialize the CarDetector class.

        Args:
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Minimum confidence required to consider a detection valid.

        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker = CentroidTracker()

    def detect_cars_for_test(self, frame: np.ndarray) -> List[BoundingBox]:
        # Extract results from the detections
        results = self.model(frame)
        detected_cars = results[0].boxes

        detections = []

        for box in detected_cars:
            confidence = box.conf[0].item()  # Get confidence
            class_id = int(box.cls[0].item())  # Get the class ID

            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_car_box = BoundingBox(x1, y1, x2 - x1, y2 - y1, confidence)

            # Class ID 2 is 'car', Class ID 3 is 'motorcycle', Class ID 5 is 'bus', Class ID 7 is 'truck' 
            is_object_of_interest = (class_id == 2 or class_id == 3 or class_id == 5 or class_id == 7)

            # Check if the detected car overlaps with any motion detection
            if confidence > self.confidence_threshold and is_object_of_interest:
                detections.append(detected_car_box)

        return detections

    def detect_moving_cars(self, visualize_frame: np.ndarray, frame: np.ndarray,
                           motion_bounding_boxes: List[BoundingBox], direction_callback) -> Dict:
        """
        Detect moving cars by checking YOLO car detections that overlap with motion detections.

        Args:
            frame (np.ndarray): The input frame.
            motion_bounding_boxes (List[BoundingBox]): List of bounding boxes for motion detections.

        Returns:
            List[BoundingBox]: List of bounding boxes for moving cars.
        """
        moving_cars = []

        results = self.model(frame)
        detected_cars = results[0].boxes
        detections = []

        for box in detected_cars:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_car_box = BoundingBox(x1, y1, x2 - x1, y2 - y1, confidence)

            is_object_of_interest = class_id in [2, 3, 5, 7]

            if confidence > self.confidence_threshold and is_object_of_interest:
                for motion_box in motion_bounding_boxes:
                    if detected_car_box.intersects_with(motion_box):
                        moving_cars.append(detected_car_box)
                        detections.append((x1, y1, x2, y2))
                        break

        filtered_detections = self.tracker.non_max_suppression_fast(detections)
        updated_objects = self.tracker.update(filtered_detections)

        # Use direction callback clearly here:
        for obj in updated_objects.values():
            bbox = obj["bbox"]
            obj["direction"] = direction_callback(bbox)

        return updated_objects


class LicensePlateDetector:

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the LicensePlateDetector class.
        
        Args:
            model_path (str): Path to the YOLO model file.
            allowed_formats: A list of regex patterns for valid plate formats.
            confidence_threshold (float): Minimum confidence required to consider a detection valid.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.reader = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
       
    def detect_license_plates(self, visualize_frame: np.ndarray, frame: np.ndarray, detected_cars: OrderedDict, is_in_entrance_or_exit):
        """
        Detect license plates in the frame and associate them with tracked cars.
        The matching logic is extracted into a separate method.
        """
        license_plates = self.model(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = license_plate
            if confidence > self.confidence_threshold and int(class_id) == 0:
                plate_box = BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1), confidence)

                # Get the best matching car for the license plate
                best_match_car_id = self.match_plate_to_car(plate_box, detected_cars)

                if best_match_car_id is not None:
                    # Retrieve the car's bounding box and details
                    car_details = detected_cars[best_match_car_id]
                    prev_plate_number = car_details["plate_number"]
                    prev_confidence = car_details["confidence"]
                    last_timestamp = car_details["last_timestamp"] 
                    occurs = car_details["occurs"]

                    difference = datetime.now() - last_timestamp
                    if difference.total_seconds() > 1 and occurs < 3:

                        # Crop the license plate region
                        cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]

                        ocr_text, ocr_confidence = self.read_text_from_plate(cropped_plate, best_match_car_id)
                        processed_text = process_plate(ocr_text) if ocr_text is not None else None
                        if (processed_text is not None and ocr_confidence > prev_confidence): 
                            car_details["direction"] = is_in_entrance_or_exit((x1, y1, x2, y2))
                            car_details['plate_number'] = processed_text
                            car_details["confidence"] = ocr_confidence
                            car_details["last_timestamp"] = datetime.now()
                            if prev_plate_number == processed_text:
                                car_details["occurs"] = occurs + 1
                            else:
                                car_details["occurs"] = 0

        return detected_cars

    def match_plate_to_car(self, plate_box: BoundingBox, detected_cars: OrderedDict) -> int:
        """
        Match a license plate to the best-fitting car based on spatial alignment and proximity.

        Args:
            plate_box (BoundingBox): The detected license plate bounding box.

        Returns:
            int: The ID of the best-matching car, or None if no match is found.
        """
        best_match_car_id = None
        smallest_distance = float('inf')

        for car_id, car_details in detected_cars.items():
            car_center = car_details["centroid"]
            car_box = car_details['bbox']

            # Calculate the center points of the car and plate boxes
            plate_center_x = plate_box.x + plate_box.width / 2
            plate_center_y = plate_box.y + plate_box.height / 2

            # Calculate Euclidean distance between the centers
            distance = ((car_center[0] - plate_center_x) ** 2 + (car_center[1] - plate_center_y) ** 2) ** 0.5

            # Ensure spatial alignment (plate is near the car's expected position)
            if self.is_spatially_aligned(car_box, plate_box) and distance < smallest_distance:
                smallest_distance = distance
                best_match_car_id = car_id

        return best_match_car_id

    def is_spatially_aligned(self, car_box: tuple, plate_box: BoundingBox) -> bool:
        """
        Check if the license plate is spatially aligned with the car.
        Assumes plates are near the bottom or rear of a car.
        """
        # Calculate the height of the car
        car_height = car_box[3] - car_box[1]  # y2 - y1
        car_bottom_y = car_box[3]  # y2 is the bottom of the car
        plate_bottom_y = plate_box.y + plate_box.height

        # Ensure the plate is vertically close to the car's bottom
        vertically_aligned = car_bottom_y - plate_bottom_y < car_height / 3

        # Ensure the plate is horizontally within the car's width
        horizontally_aligned = (
            plate_box.x > car_box[0] and  # x1 of car
            plate_box.x + plate_box.width < car_box[2]  # x2 of car
        )

        return vertically_aligned and horizontally_aligned

    def read_text_from_plate(self, cropped_plate: np.ndarray, car_id,confidence_threshold = 0.8) -> str:
        """
        Read text from the cropped license plate using OCR, with filtering for irrelevant areas.

        Args:
            cropped_plate (np.ndarray): Cropped image of the license plate.

        Returns:
            str: Validated license plate text.
        """
        # Resize the plate to standard size
        resized = util.resize_plate(cropped_plate)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Remove noise            
        blur = cv2.GaussianBlur(gray, (3,3), 0)

        # Sharpen using HBF
        sharp = util.sharpenHBF(blur)

        # Clip values to maintain valid range
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        ocr_ready = sharp

        # cv2.imshow(f"ocr-plate {car_id}", ocr_ready)

        # Use OCR
        try:
            results = self.reader.ocr(ocr_ready)

            selected_result = None
            selected_confidence = 0.0

            # Check if results is None
            if not results or results[0] is None:
                return None, 0.0
            
            # Extract text with high confidence
            high_confidence_results = [
                (line[1][0], line[1][1]) for line in results[0] if line[1][1] >= confidence_threshold
            ]

            if high_confidence_results:
                # Select the result with the highest confidence
                selected_result, selected_confidence = max(high_confidence_results, key=lambda x: x[1])
            return selected_result, selected_confidence

        except Exception as e:
            print(f"Error during OCR: {e}")
            return None, 0.0