import os
import json
import cv2
import sys

from typing import List, Tuple
# Dynamically add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from edge import BoundingBox
from cloud import CarDetector


def load_ground_truth(annotation_file: str) -> List[dict]:
    """
    Load ground truth bounding boxes from the annotation file.
    """
    if not os.path.exists(annotation_file):
        return []
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
        return [anno["rectMask"] for anno in annotations]


def evaluate_model(detected_frames: List[List[dict]], gt_frames: List[List[dict]]) -> Tuple[float, float, float]:
    """
    Evaluate the detection results using precision, recall, and F1-score.
    """
    # Placeholder for evaluation logic; replace with actual implementation.
    precision = recall = f1 = 1.0  # Example values
    return precision, recall, f1


def test_car_detection(model: CarDetector, frames_path: str, annotations_path: str):
    """
    Test the car detection model on a set of frames and annotations.
    """
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png'))])
    detected_frames = []
    gt_frames = []

    for frame_file in frame_files:
        frame_path = os.path.join(frames_path, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: Unable to read frame {frame_path}")
            continue

        # Detect cars in the frame
        motion_bboxes = []  # Replace with actual motion bounding boxes if applicable
        detected_bboxes = model.detect_moving_cars(frame, frame, motion_bboxes)
        detected_frames.append(detected_bboxes)

        # Load corresponding ground truth annotations
        annotation_file = os.path.join(
            annotations_path, frame_file.replace('.jpg', '.json').replace('.png', '.json')
        )
        gt_bboxes = load_ground_truth(annotation_file)
        gt_frames.append(gt_bboxes)

    # Evaluate the model
    precision, recall, f1 = evaluate_model(detected_frames, gt_frames)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")


if __name__ == "__main__":
    # Configure the car detector and paths
    detector = CarDetector("yolo11n.pt")
    video_name = "motion3"
    frames_path = f"tests/frames/{video_name}"
    annotations_path = f"tests/annotations/{video_name}"

    # Run the test
    test_car_detection(detector, frames_path, annotations_path)
