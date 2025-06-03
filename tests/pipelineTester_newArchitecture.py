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



class pipelineTesterMicroservice:
    def __init__(self, video_path: str, ground_truth: List[Dict],
                 partial_match_threshold: float = 0.75,
                 check_direction: bool = True,
                 check_authorization: bool = True):
        self.video_path = video_path
        self.ground_truth = [
            {"plate": process_plate(gt["plate"].upper()),
             "direction": gt.get("direction"),
             "authorized": gt.get("authorized")}
            for gt in ground_truth if gt.get("plate")
        ]
        self.partial_match_threshold = partial_match_threshold
        self.check_direction = check_direction
        self.check_authorization = check_authorization
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
        from collections import Counter, defaultdict
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
        correct_dirs = 0
        correct_auth = 0
        direction_comparisons = []
        auth_comparisons = []

        for pred in detected:
            plate = pred["plate"]
            actual_dir = pred.get("direction")
            actual_auth = pred.get("authorized")
            matched = False

            if gt_counts[plate] > 0:
                exact_matches.append(plate)
                gt_counts[plate] -= 1
                tp += 1
                matched = True

                # Check direction and authorization for exact matches
                expected_dir = next((gt["direction"] for gt in self.ground_truth if gt["plate"] == plate), None)
                expected_auth = next((gt["authorized"] for gt in self.ground_truth if gt["plate"] == plate), None)

                direction_comparisons.append((plate, expected_dir, actual_dir))
                auth_comparisons.append((plate, expected_auth, actual_auth))

                if self.check_direction and expected_dir == actual_dir:
                    correct_dirs += 1

                if self.check_authorization and expected_auth is not None and expected_auth == actual_auth:
                    correct_auth += 1

            else:
                # Try partial match
                best_match = None
                best_ratio = 0.0
                best_gt = None
                for gt in self.ground_truth:
                    ratio = difflib.SequenceMatcher(None, plate or "", gt["plate"] or "").ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = gt["plate"]
                        best_gt = gt

                if best_ratio >= 0.6:  # Lower threshold for logging
                    partial_matches.append((plate, best_match, best_ratio))
                    if best_gt:
                        direction_comparisons.append((plate, best_gt.get("direction"), actual_dir,
                                                   f"Partial match: {best_match} ({best_ratio:.2f})"))
                        auth_comparisons.append((plate, best_gt.get("authorized"), actual_auth,
                                              f"Partial match: {best_match} ({best_ratio:.2f})"))

                if best_ratio >= self.partial_match_threshold and gt_counts[best_match] > 0:
                    gt_counts[best_match] -= 1
                    tp += 1
                else:
                    unmatched.append(plate)
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
                match_type = "ACCEPTED" if score >= self.partial_match_threshold else "REJECTED"
                print(f"Detected: {pred} | GT: {gt} | Similarity: {score:.2f} | {match_type}")

        if unmatched:
            print("\n==================== UNMATCHED DETECTIONS ==============")
            for plate in unmatched:
                print(plate)

        if fn > 0:
            print("\n==================== MISSING PLATES =====================")
            for plate, count in gt_counts.items():
                if count > 0:
                    print(f"{plate} (missed {count}x)")

        # if self.check_direction:
        #     print("\n==================== DIRECTION DETECTION ===================")
        #     print(f"Direction Accuracy: {correct_dirs}/{tp} = {correct_dirs/tp if tp > 0 else 0:.2f}")
        #     print("\nDirection Comparison per Plate:")
        #     for plate, expected, actual, note in direction_comparisons:
        #         print(f"Plate: {plate} | Expected: {expected} | Detected: {actual} | {note}")

        # if self.check_authorization:
        #     print("\n==================== AUTHORIZATION DETECTION ===================")
        #     print(f"Authorization Accuracy: {correct_auth}/{tp} = {correct_auth/tp if tp > 0 else 0:.2f}")
        #     print("\nAuthorization Comparison per Plate:")
        #     for plate, expected, actual, note in auth_comparisons:
        #         print(f"Plate: {plate} | Expected: {expected} | Detected: {actual} | {note}")

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
        {"plate": "43788503", "direction": "Exit", "authorized": 0}
        # {"plate": "80304001", "direction": "Entrance", "authorized": 0},
        # {"plate": "59476603", "direction": "Exit", "authorized": 0},
        # {"plate": "73778302", "direction": "Exit", "authorized": 0},
        # {"plate": "88731303", "direction": "Exit", "authorized": 0}

    ]

    tester = pipelineTesterMicroservice("recordings/short.mp4", ground_truth)
    tester.run_test()
