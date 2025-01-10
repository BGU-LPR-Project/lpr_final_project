import cv2
import numpy as np
from typing import List
from ultralytics import YOLO
import torch
from AI1 import BoundingBox
import paddleocr

class LicensePlateDetector:

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the LicensePlateDetector class.
        
        Args:
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Minimum confidence required to consider a detection valid.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.reader = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
       
    def detect_license_plates(self, frame: np.ndarray, detected_cars: List[BoundingBox]):
        """
        Detect license plates in the frame and associate them with detected cars.
        Draw bounding boxes for cars and plates, and display OCR results above the car bounding boxes.
        Only consider plates fully contained within a car's bounding box.
        """
        license_plates = self.model(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = license_plate
            if confidence > self.confidence_threshold and int(class_id) == 0:
                plate_box = BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1), confidence)

                # Check if the plate is fully contained within any car bounding box
                for car_box in detected_cars:
                    if (plate_box.x >= car_box.x and
                        plate_box.y >= car_box.y and
                        plate_box.x + plate_box.width <= car_box.x + car_box.width and
                        plate_box.y + plate_box.height <= car_box.y + car_box.height):
                        
                        # Crop the license plate region
                        cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]

                        # Perform OCR on the cropped plate
                        ocr_text = self.read_text_from_plate(cropped_plate)

                        # Draw bounding box for the car
                        cv2.rectangle(frame, (car_box.x, car_box.y), (car_box.x + car_box.width, car_box.y + car_box.height), (255, 0, 0), 2)
                        
                        # Display OCR result above the car bounding box
                        car_label = f"Car: {ocr_text}"
                        # Determine text size
                        (text_width, text_height), baseline = cv2.getTextSize(car_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        # Draw a background rectangle for better readability
                        label_x1 = car_box.x
                        label_y1 = car_box.y - 10 - text_height - baseline
                        label_x2 = car_box.x + text_width
                        label_y2 = car_box.y - 10
                        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (255, 0, 0), -1)

                        # Put text on top of the car bounding box
                        cv2.putText(frame, car_label, (car_box.x, car_box.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Draw bounding box for the license plate
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                        plate_label = f"Conf: {confidence:.2f}"
                        cv2.putText(frame, plate_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


    def read_text_from_plate(self, cropped_plate: np.ndarray) -> str:
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
            # Set a confidence threshold (e.g., 0.8)
            confidence_threshold = 0.8

            # Extract text with high confidence
            high_confidence_results = [
                line[1][0] for line in results[0] if line[1][1] >= confidence_threshold
            ]
            
            return high_confidence_results

        except Exception as e:
            print(f"Error dur")