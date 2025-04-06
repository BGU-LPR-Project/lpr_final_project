import sys
import os

# Dynamically add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from main import LPRPipeline
from formats import process_plate
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import difflib

class pipelineTester:
    def __init__(self, video_path: str, ground_truth: List[Dict],
                 check_direction: bool = True,
                 check_tracking: bool = True,
                 partial_match_threshold: float = 0.9):
        """
        ground_truth: List of dictionaries with 'plate' and optional 'direction'
        e.g., [{"plate": "MH01EB2570", "direction": "Entrance"}, ...]
        """
        self.video_path = video_path
        self.ground_truth = [
            {
                "plate": process_plate(gt["plate"].upper()),
                "direction": gt.get("direction")
            } for gt in ground_truth if gt.get("plate")
        ]
        self.check_direction = check_direction
        self.check_tracking = check_tracking
        self.partial_match_threshold = partial_match_threshold

    def run_test(self) -> Tuple[float, float, float]:
        pipeline = LPRPipeline(self.video_path)
        pipeline.paused = False

        # Monkey-patch UI methods to suppress rendering
        pipeline.visualize = lambda *args, **kwargs: None

        pipeline.run()

        # Collect predictions
        detected_plates = [
            {
                "plate": process_plate(p.plate_number.upper()),
                "direction": p.direction,
                "object_id": pid
            } for pid, p in pipeline.license_plate_table.items()
            if p.plate_number and p.plate_number != '---'
        ]

        return self.evaluate(detected_plates)

    def evaluate(self, detected: List[Dict]) -> Tuple[float, float, float]:
        gt_counts = Counter([d["plate"] for d in self.ground_truth])
        detected_counts = Counter([d["plate"] for d in detected])

        tp, fp, fn = 0, 0, 0
        correct_dirs = 0
        id_map = defaultdict(set)
        partial_matches = []

        for pred in detected:
            plate = pred["plate"]
            matched = False

            if plate in gt_counts and gt_counts[plate] > 0:
                tp += 1
                gt_counts[plate] -= 1
                matched = True

                if self.check_direction:
                    for gt in self.ground_truth:
                        if gt["plate"] == plate and gt["direction"] == pred["direction"]:
                            correct_dirs += 1
                            break

                if self.check_tracking:
                    id_map[plate].add(pred["object_id"])
            else:
                best_match = None
                best_ratio = 0.0
                for gt in self.ground_truth:
                    ratio = difflib.SequenceMatcher(None, plate or "", gt["plate"] or "").ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = gt["plate"]

                if best_ratio >= 0.6:
                    partial_matches.append((plate, best_match, best_ratio))

                if best_ratio >= self.partial_match_threshold:
                    tp += 1
                else:
                    fp += 1

        fn = sum(gt_counts.values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print("\n--- Evaluation Summary ---")
        print(f"Detected Plates: {[d['plate'] for d in detected]}")
        print(f"Ground Truth Plates: {[d['plate'] for d in self.ground_truth]}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        if self.check_direction:
            print(f"Direction Accuracy: {correct_dirs}/{tp} = {correct_dirs / tp if tp > 0 else 0:.2f}")

        if self.check_tracking:
            consistent = sum(len(ids) == 1 for ids in id_map.values())
            print(f"Tracking Consistency: {consistent}/{len(id_map)} unique IDs per plate")

        if partial_matches:
            print("\nPartial Matches:")
            for det, gt, score in partial_matches:
                print(f"Detected: {det} | Closest GT: {gt} | Similarity: {score:.2f}")

        return precision, recall, f1


if __name__ == "__main__":
    video_path = "recordings/rec6.mp4"
    ground_truth = [
        {"plate": "MH01EB2570", "direction": "Entrance"},
        {"plate": "MH02FX4729", "direction": "Exit"},
        {"plate": "MH04HU1278", "direction": "Entrance"},
        {"plate": "MH48AC4033", "direction": "Exit"},
    ]

    tester = pipelineTester(video_path, ground_truth)
    tester.run_test()
