
import sys
import os



# Dynamically add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import cv2
from main import LPRPipeline
from formats import process_plate
from typing import List, Tuple
from collections import Counter
import difflib


class FullPipelineTester:
    def __init__(self, video_path: str, ground_truth_plates: List[str]):
        self.video_path = video_path
        self.ground_truth = [process_plate(p.upper()) for p in ground_truth_plates if p]

    def run_test(self) -> Tuple[float, float, float]:
        pipeline = LPRPipeline(self.video_path)
        pipeline.paused = False

        # Monkey-patch UI methods to avoid rendering in test mode
        pipeline.visualize = lambda *args, **kwargs: None

        # Run the pipeline
        pipeline.run()

        # Gather detected plates
        detected_plates = [
            process_plate(p.plate_number.upper())
            for p in pipeline.license_plate_table.values()
            if p.plate_number and p.plate_number != '---'
        ]

        return self.evaluate(detected_plates)

    def evaluate(self, detected: List[str]) -> Tuple[float, float, float]:
        gt_counts = Counter(self.ground_truth)
        detected_counts = Counter(detected)

        tp = 0
        fp = 0
        fn = 0

        partial_matches = []

        matched_gt = set()

        for plate in detected:
            if plate in gt_counts and gt_counts[plate] > 0:
                tp += 1
                gt_counts[plate] -= 1
            else:
                # Check for partial match using similarity ratio
                best_match = None
                best_ratio = 0.0
                for gt in gt_counts:
                    ratio = difflib.SequenceMatcher(None, plate or "", gt or "").ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = gt

                if best_ratio >= 0.6:
                    partial_matches.append((plate, best_match, best_ratio))
                fp += 1


        # Remaining in ground truth are false negatives
        fn = sum(gt_counts.values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print("\nEvaluation Summary:")
        print(f"Detected Plates: {detected}")
        print(f"Ground Truth Plates: {self.ground_truth}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        if partial_matches:
            print("\nPartial Matches:")
            for det, gt, score in partial_matches:
                print(f"Detected: {det} | Closest GT: {gt} | Similarity: {score:.2f}")
#maybe add to be tp if its a partial match over 0.90

        return precision, recall, f1



if __name__ == "__main__":

    video_path = "recordings\\rec6.mp4"
    expected_plates = ["MH01EB2570", "MH02FX4729", "MHO4HU1278", "MH48AC4033"]

    tester = FullPipelineTester(video_path, expected_plates)
    precision, recall, f1 = tester.run_test()
