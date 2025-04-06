import os
import json
import cv2
import sys
import numpy as np
from typing import List

# Dynamically add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from edge_service.edge import MotionDetector, BoundingBox
from cloud_service.cloud import CarDetector


def test_motion_detection_with_annotations(model, frames_path, annotations_path, task):
    # Get all frame filenames
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png'))])
    detected_frames = []
    gt_frames = []

    for frame_file in frame_files:
        # Read the frame
        frame_path = os.path.join(frames_path, frame_file)
        frame = cv2.imread(frame_path)
        
        # Detect motion in the frame
        detected_bboxes = model.detect(frame)
        detected_frames.append(detected_bboxes)
        
        # Load the corresponding annotation file
        annotation_file = os.path.join(annotations_path, frame_file.replace('.jpg', '.json').replace('.png', '.json'))
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
                gt_bboxes = [anno["rectMask"] for anno in annotations]
        else:
            # If no annotation exists, assume no ground truth bounding boxes
            gt_bboxes = []
        
        gt_frames.append(gt_bboxes)

        for det in detected_bboxes:
            b1 = (int(det.x), int(det.y), int(det.x + det.width), int(det.y + det.height))
            cv2.rectangle(frame, (b1[0], b1[1]), (b1[2], b1[3]), (255, 0, 0), 1)

        for gt in gt_bboxes:
            b1 = (int(gt["xMin"]), int(gt["yMin"]), int(gt["xMin"] + gt["width"]), int(gt["yMin"] + gt["height"]))
            cv2.rectangle(frame, (b1[0], b1[1]), (b1[2], b1[3]),(0, 255, 255), 1)
                          
        cv2.imshow("Test", frame)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('n'):
            continue
    # Evaluate the model
    precision, recall, f1 = evaluate_model(detected_frames, gt_frames)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

def evaluate_model(detected_frames, gt_frames):
    tp = 0  # true positives
    fp = 0  # false positives
    fn = 0  # false negatives

    for detected_bboxes, gt_bboxes in zip(detected_frames, gt_frames):
        for gt in gt_bboxes:
            matched = False
            for det in detected_bboxes:
                b1 = (int(det.x), int(det.y), int(det.x + det.width), int(det.y + det.height))
                b2 = (int(gt["xMin"]), int(gt["yMin"]), int(gt["xMin"] + gt["width"]), int(gt["yMin"] + gt["height"]))
                if (int(det.x) <= int(gt["xMin"]) and int(det.y) <= int(gt["yMin"]) and
                    int(det.x + det.width) >= int(gt["xMin"] + gt["width"]) and
                    int(det.y + det.height) >= int(gt["yMin"] + gt["height"])) or calculate_overlap_percentage(b1, b2) >= 70:
                        tp += 1
                        matched = True
                        break

            if not matched:
                fn += 1

        # Count false positives for any detected bbox without matching ground truth
        for det in detected_bboxes:
            matched = False
            for gt in gt_bboxes:
                b1 = (int(det.x), int(det.y), int(det.x + det.width), int(det.y + det.height))
                b2 = (int(gt["xMin"]), int(gt["yMin"]), int(gt["xMin"] + gt["width"]), int(gt["yMin"] + gt["height"]))
                if (int(det.x) <= int(gt["xMin"]) and int(det.y) <= int(gt["yMin"]) and
                    int(det.x + det.width) >= int(gt["xMin"] + gt["width"]) and
                    int(det.y + det.height) >= int(gt["yMin"] + gt["height"])) or calculate_overlap_percentage(b1, b2) >= 70:
                        matched = True
                        break

            if not matched:
                fp += 1

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def calculate_overlap_percentage(box1, box2):
  """
  Calculates the percentage of overlap between two bounding boxes.

  Args:
    box1: A tuple or list representing the first bounding box in the format (x1, y1, x2, y2).
    box2: A tuple or list representing the second bounding box in the format (x1, y1, x2, y2).

  Returns:
    The percentage of overlap between the two boxes (float between 0 and 100).
  """

  # Extract coordinates
  x1_min, y1_min, x1_max, y1_max = box1
  x2_min, y2_min, x2_max, y2_max = box2

  # Calculate intersection coordinates
  x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
  y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

  # Calculate intersection area
  intersection_area = x_overlap * y_overlap

  # Calculate union area
  area1 = (x1_max - x1_min) * (y1_max - y1_min)
  area2 = (x2_max - x2_min) * (y2_max - y2_min)
  union_area = area1 + area2 - intersection_area

  # Calculate overlap percentage
  if union_area == 0:
    return 0  # Avoid division by zero
  overlap_percentage = (intersection_area / union_area) * 100

  return overlap_percentage


class BaseDetector:
    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        return None
    
class Motion(BaseDetector):
    def __init__(self):
        super().__init__()
        self.model = MotionDetector()

    def detect(self, frame) -> List[BoundingBox]:
        return self.model.detect_motion(frame, frame)
    
class Vehicle(BaseDetector):
    def __init__(self):
        super().__init__()
        self.model = CarDetector("yolo11n.pt")

    def detect(self, frame) -> List[BoundingBox]:
        return self.model.detect_cars_for_test(frame)
    

detector = Vehicle()
video_name = "car2"
test_motion_detection_with_annotations(detector, f"tests\\frames\\{video_name}", f"tests\\annotations\\{video_name}", task="car_detection")