import cv2
import numpy as np
from typing import List
from ultralytics import YOLO

class BoundingBox:
    def __init__(self, x: int, y: int, width: int, height: int, confidence: float = 1.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence

    def intersects_with(self, other_box: 'BoundingBox') -> bool:
        """
        Check if this box intersects with another.
        """
        return not (
            self.x + self.width < other_box.x or
            self.x > other_box.x + other_box.width or
            self.y + self.height < other_box.y or
            self.y > other_box.y + other_box.height
        )

    def area(self) -> int:
        return self.width * self.height

class MotionDetector:
    def __init__(self):
        """
        Initialize the MotionDetector class
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    def detect_motion(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detects moving objects in the given frame.
        
        Args:
            frame (np.ndarray): The input frame (BGR image).
            
        Returns:
            List[BoundingBox]: List of bounding boxes for detected motion regions.
        """
        # Apply the background subtractor to get the foreground mask
        fg_mask = self.bg_subtractor.apply(frame)

        # Optionally remove noise from the foreground mask
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # Clean the mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            # Filter small contours based on area
            if cv2.contourArea(contour) < 900:  # Threshold to ignore small movements
                continue

            # Get the bounding box for each valid contour
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append(BoundingBox(x, y, w, h))

            # Draw bounding box for the movement
            # cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 255), 1)

        return bounding_boxes
    

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

    def detect_moving_cars(self, frame: np.ndarray, motion_bounding_boxes: List[BoundingBox]) -> List[BoundingBox]:
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

                        # Draw bounding box for the moving car
                        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                        # plate_label = f"car: {confidence:.2f}"
                        # cv2.putText(frame, plate_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        continue
                       
        return moving_cars