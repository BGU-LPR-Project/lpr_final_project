import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import redis
import pickle
import time
import cv2
from typing import List, Dict, Tuple
import difflib
from video_service.video_handler import VideoHandler
from cloud_service.formats import process_plate

# Removed the misplaced line as it is redundant and already defined in the class constructor.


class pipelineTesterMicroservice:
    def __init__(self, video_path: str, ground_truth: List[Dict], partial_match_threshold: float = 0.9):
        self.video_path = video_path
        self.ground_truth = [
            {"plate": process_plate(gt["plate"].upper()), "direction": gt.get("direction"), "authorized": gt.get("authorized")}
            for gt in ground_truth if gt.get("plate")
        ]
        self.partial_match_threshold = partial_match_threshold
        self.redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.redis_client.flushall()

    def push_frames(self):
        handler = VideoHandler(self.video_path, target_fps=4)
        handler.load_video()

        timeout = 30
        last_frame_time = time.time()

        while True:
            frame = handler.decode_frame()
            if frame is None:
                if time.time() - last_frame_time > timeout:
                    break
                time.sleep(0.1)
                continue
            last_frame_time = time.time()

            frame_data = pickle.dumps(frame)
            self.redis_client.rpush("frame_queue", frame_data)
            time.sleep(0.2)

        handler.release_resources()

    def wait_for_results(self, timeout=15):
        print("Waiting for results...")
        start_time = time.time()
        results = []

        while time.time() - start_time < timeout:
            try:
                data = self.redis_client.get("tracked_plates")
                if data:
                    results = pickle.loads(data)
                    break
            except Exception as e:
                print(f"Error while fetching results: {e}")
            time.sleep(1)

        return results

    def evaluate(self, detected: List[Dict]) -> Tuple[float, float, float]:
        from collections import Counter
        import difflib

        gt_plates = [d["plate"] for d in self.ground_truth]
        pred_plates = [d["plate"] for d in detected]

        print("\n===================== DETECTED PLATES =====================")
        for plate in pred_plates:
            print(plate)

        print("\n===================== GROUND TRUTH PLATES =================")
        for plate in gt_plates:
            print(plate)

        gt_counts = Counter(gt_plates)
        tp, fp, fn = 0, 0, 0
        exact_matches = []
        partial_matches = []
        unmatched = []

        for pred in pred_plates:
            if gt_counts[pred] > 0:
                exact_matches.append(pred)
                gt_counts[pred] -= 1
                tp += 1
            else:
                # Try partial match
                best_match = None
                best_ratio = 0.0
                for gt in gt_plates:
                    ratio = difflib.SequenceMatcher(None, pred or "", gt or "").ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = gt
                if best_ratio >= self.partial_match_threshold and gt_counts[best_match] > 0:
                    partial_matches.append((pred, best_match, best_ratio))
                    gt_counts[best_match] -= 1
                    tp += 1
                else:
                    unmatched.append(pred)
                    fp += 1

        # Count remaining unmatched GT
        fn = sum(gt_counts.values())

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        print("\n==================== MATCHED PLATES ====================")
        for plate in exact_matches:
            print(plate)

        if partial_matches:
            print("\n==================== PARTIAL MATCHES ===================")
            for pred, gt, score in partial_matches:
                print(f"Detected: {pred} | GT: {gt} | Similarity: {score:.2f}")

        if unmatched:
            print("\n==================== UNMATCHED DETECTIONS ==============")
            for plate in unmatched:
                print(plate)

        if fn > 0:
            print("\n==================== MISSING PLATES =====================")
            for plate, count in gt_counts.items():
                if count > 0:
                    print(f"{plate} (missed {count}x)")

        print("\n====================== METRICS ==========================")
        print(f"TP: {tp}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print("=========================================================")
        return precision, recall, f1

    
    def run_test(self):
        self.push_frames()
        detected_raw = self.wait_for_results()

        # Example detected format: [(id, plate, conf)]
        detected = [{"plate": process_plate(p.upper())} for _, p, _ in detected_raw if p and p != '---']
        return self.evaluate(detected)


if __name__ == "__main__":
    ground_truth = [
        {"plate": "MH01EB2570", "direction": "Exit", "authorized": 0},
        {"plate": "MH02FX4729", "direction": "Entrance", "authorized": 0},
        {"plate": "MH04HU1278", "direction": "Exit", "authorized": 0},
        {"plate": "MH48AC4033", "direction": "Entrance", "authorized": 0},
    ]

    tester = pipelineTesterMicroservice("recordings/rec6.mp4", ground_truth)
    tester.run_test()
