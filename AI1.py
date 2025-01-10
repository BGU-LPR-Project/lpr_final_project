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

class MotionDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)
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
        
        merged_boxes = self.merge_boxes(bounding_boxes)

        for box in merged_boxes:
            # Draw bounding box for the movement
            cv2.rectangle(frame, (box.x, box.y), (box.x + box.width, box.y + box.height), (0, 255, 255), 1)

        return merged_boxes

    def merge_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        if not boxes:
            return []
        boxes.sort(key=lambda b: (b.y, b.x))
        merged = [boxes[0]]
        for current in boxes[1:]:
            last = merged[-1]
            if self.should_merge(last, current):
                merged[-1] = last.merge_with(current)
            else:
                merged.append(current)
        return merged

    def should_merge(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        # If no intersection, check closeness
        threshold = min(box1.width, box1.height, box2.width, box2.height) / 4
        close_in_x = abs(box1.x + box1.width / 2 - (box2.x + box2.width / 2)) < threshold
        close_in_y = abs(box1.y + box1.height / 2 - (box2.y + box2.height / 2)) < threshold

        # Use IoU as an additional condition
        return box1.intersects_with(box2) or (close_in_x and close_in_y and self.intersect_over_union(box1, box2) < 0.7)


    def intersect_over_union(self, box1: BoundingBox, box2: BoundingBox):
        inter_x1 = max(box1.x, box2.x)
        inter_y1 = max(box1.y, box2.y)
        inter_x2 = min(box1.x + box1.width, box2.x + box2.width)
        inter_y2 = min(box1.y + box1.height, box2.y + box2.height)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = box1.width * box1.height
        box2_area = box2.width * box2.height
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0



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
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                        plate_label = f"car: {confidence:.2f}"
                        cv2.putText(frame, plate_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        continue

        return moving_cars