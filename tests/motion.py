import os
import json
import cv2
import sys

# Dynamically add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from edge import MotionDetector

def test_motion_detection_with_annotations(model, frames_path, annotations_path):
    # Get all frame filenames
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png'))])
    detected_frames = []
    gt_frames = []

    for frame_file in frame_files:
        # Read the frame
        frame_path = os.path.join(frames_path, frame_file)
        frame = cv2.imread(frame_path)
        
        # Detect motion in the frame
        detected_bboxes = model.detect_motion(frame)
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
                if (det.x <= gt["xMin"] and det.y <= gt["yMin"] and
                    det.x + det.width >= gt["xMin"] + gt["width"] and
                    det.y + det.height >= gt["yMin"] + gt["height"]):
                    tp += 1
                    matched = True
                    break

            if not matched:
                fn += 1

        # Count false positives for any detected bbox without matching ground truth
        for det in detected_bboxes:
            matched = False
            for gt in gt_bboxes:
                if (det.x <= gt["xMin"] and det.y <= gt["yMin"] and
                    det.x + det.width >= gt["xMin"] + gt["width"] and
                    det.y + det.height >= gt["yMin"] + gt["height"]):
                    matched = True
                    break

            if not matched:
                fp += 1

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

detector = MotionDetector()
video_name = "motion3"
test_motion_detection_with_annotations(detector, f"tests\\frames\\{video_name}", f"tests\\annotations\\{video_name}")