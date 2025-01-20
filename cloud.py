import cv2
import numpy as np
from typing import List
from ultralytics import YOLO
from edge import BoundingBox
import paddleocr
from tracking import CentroidTracker
from collections import OrderedDict
import re


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

    def detect_moving_cars(self, visualize_frame: np.ndarray, frame: np.ndarray, motion_bounding_boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Detect moving cars by checking YOLO car detections that overlap with motion detections.

        Args:
            frame (np.ndarray): The input frame.
            motion_bounding_boxes (List[BoundingBox]): List of bounding boxes for motion detections.

        Returns:
            List[BoundingBox]: List of bounding boxes for moving cars.
        """
        moving_cars = []

        # Perform detection using the YOLO model
        results = self.model(frame)

        # Extract results from the detections
        detected_cars = results[0].boxes

        detections = []

        for box in detected_cars:
            confidence = box.conf[0].item()  # Get confidence
            class_id = int(box.cls[0].item())  # Get the class ID

            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_car_box = BoundingBox(x1, y1, x2 - x1, y2 - y1, confidence)

            # Check if the detected car overlaps with any motion detection
            if confidence > self.confidence_threshold and class_id == 2:  # Class ID 2 is 'car'
                for motion_box in motion_bounding_boxes:
                    if detected_car_box.intersects_with(motion_box):
                        moving_cars.append(detected_car_box)
                        detections.append((x1,y1,x2,y2))
                        continue

        filtered_detections = self.tracker.non_max_suppression_fast(detections)
        cars = self.tracker.update(filtered_detections)

        return cars

class LicensePlateDetector:

    def __init__(self, model_path: str, allowed_formats: list[str], confidence_threshold: float = 0.5):
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
        self.allowed_formats = [re.compile(fmt) for fmt in allowed_formats]
       
    def detect_license_plates(self, visualize_frame: np.ndarray, frame: np.ndarray, detected_cars: OrderedDict):
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
                    current_plate_number = car_details["plate_number"]
                    current_plate_confidence = car_details["confidence"]
                    # Crop the license plate region
                    cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]
                    ocr_text, ocr_confidence = self.read_text_from_plate(cropped_plate)
                    processed_text = self.process_plate(ocr_text) if ocr_text is not None else None
                    if (processed_text is not None) and (ocr_confidence > current_plate_confidence): 
                        car_details['plate_number'] = processed_text
                        car_details["confidence"] = ocr_confidence

        for object_id, data in detected_cars.items():
            centroid = data["centroid"]
            bbox = data["bbox"]
            plate_number = data["plate_number"]

            # Draw the bounding box and centroid
            x1, y1, x2, y2 = bbox
            cv2.rectangle(visualize_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(visualize_frame, tuple(centroid), 5, (0, 255, 0), -1)
            text = f"ID {object_id}"
            cv2.putText(visualize_frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Optionally display the plate number if available
            if plate_number:
                cv2.putText(visualize_frame, f"Plate: {plate_number}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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



    def read_text_from_plate(self, cropped_plate: np.ndarray, confidence_threshold = 0.8) -> str:
        """
        Read text from the cropped license plate using OCR, with filtering for irrelevant areas.

        Args:
            cropped_plate (np.ndarray): Cropped image of the license plate.

        Returns:
            str: Validated license plate text.
        """
        # Resize for consistency
        resized = cv2.resize(cropped_plate.copy(), (300, 100), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(5,5), 0)

        # Adaptive thresholding
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # Apply a mask to focus on the central region
        h, w = otsu_thresh.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (int(0.1 * w), int(0.2 * h)), (int(0.9 * w), int(0.8 * h)), 255, -1)
        masked_plate = cv2.bitwise_and(otsu_thresh, otsu_thresh, mask=mask)
        
        cv2.imshow("plate", blurred)
        # Use OCR
        try:
            results = self.reader.ocr(blurred)

            selected_result = None
            selected_confidence = 0.0

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
        
    def validate_plate(self, plate_text: str) -> bool:
        """
        Validate the OCR result against allowed formats.
        
        Parameters:
        - plate_text: The license plate text detected by OCR.
        
        Returns:
        - bool: True if the plate matches any allowed format, False otherwise.
        """
        for pattern in self.allowed_formats:
            if re.fullmatch(pattern, plate_text):
                return True
        return False

    def format_plate(self, plate_text: str) -> str:
        """
        Format the OCR result by correcting common mistakes and standardizing.
        
        Parameters:
            plate_text: The license plate text detected by OCR.
        
        Returns:
            str: A formatted plate string.
        """
        corrections = {
            'O': '0',  # Replace letter 'O' with zero
            'I': '1',  # Replace letter 'I' with one
            'Z': '2',   # Replace letter 'Z' with two
            'S': '5',   # Replace letter 'S' with two
        }

        # Apply corrections
        corrected_plate = ''.join(corrections.get(char, char) for char in plate_text)

        return corrected_plate

    def process_plate(self, plate_text: str) -> str:
        """
        Validate and format the OCR result.
        
        Parameters:
            plate_text: The license plate text detected by OCR.
        
        Returns:
            str: The formatted plate if valid, otherwise None.
        """
        plate_text = plate_text.upper().strip()

        # Step 1: Validate the raw OCR result
        if self.validate_plate(plate_text):
            return plate_text

        # Step 2: Apply corrections and revalidate
        formatted_plate = self.format_plate(plate_text)
        if self.validate_plate(formatted_plate):
            return formatted_plate

        # If still invalid, return None
        return None