import threading
import cv2
import zmq
import base64
import pickle
import numpy as np
from collections import OrderedDict
from datetime import datetime
from ultralytics import YOLO
from scipy.spatial import distance as dist
import re
import json
import os

def load_roi_regions(path="roi_config.json"):
    if not os.path.exists(path):
        print("[EDGE] ROI config not found.")
        return {"entrance": [], "exit": []}
    with open(path, "r") as f:
        data = json.load(f)
    return {"entrance": data.get("entrance", []), "exit": data.get("exit", [])}

def get_direction_for_bbox(bbox, roi_data):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    for x1, y1, x2, y2 in roi_data.get("entrance", []):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return "Entrance"
    for x1, y1, x2, y2 in roi_data.get("exit", []):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return "Exit"
    return "Unknown"

class BoundingBox:
    def __init__(self, x, y, width, height, confidence=1.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence

    def intersects_with(self, other):
        return not (
            self.x + self.width < other.x or
            self.x > other.x + other.width or
            self.y + self.height < other.y or
            self.y > other.y + other.height
        )

class CentroidTracker:
    def __init__(self, max_disappeared=5):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "plate_number": "---",
            "direction": None,
            "confidence": 0.0,
            "last_timestamp": datetime.now(),
            "occurs": 0,
            "confirmed": False,
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detections):
            c_x = int((x1 + x2) / 2.0)
            c_y = int((y1 + y2) / 2.0)
            input_centroids[i] = (c_x, c_y)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], detections[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj["centroid"] for obj in self.objects.values()]
            D = dist.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id]["centroid"] = input_centroids[col]
                self.objects[object_id]["bbox"] = detections[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], detections[col])

        return self.objects

def decode_frame(encoded_str):
    buffer = base64.b64decode(encoded_str)
    np_arr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def is_spatially_aligned(car_box, plate_box):
    car_height = car_box[3] - car_box[1]
    car_bottom_y = car_box[3]
    plate_bottom_y = plate_box.y + plate_box.height
    vertically_aligned = car_bottom_y - plate_bottom_y < car_height / 3
    horizontally_aligned = (
        plate_box.x > car_box[0] and plate_box.x + plate_box.width < car_box[2]
    )
    return vertically_aligned and horizontally_aligned

def match_plate_to_car(plate_box, detected_cars):
    best_match_car_id = None
    smallest_distance = float("inf")
    for car_id, car_details in detected_cars.items():
        car_center = car_details["centroid"]
        car_box = car_details["bbox"]
        plate_center_x = plate_box.x + plate_box.width / 2
        plate_center_y = plate_box.y + plate_box.height / 2
        distance = ((car_center[0] - plate_center_x) ** 2 + (car_center[1] - plate_center_y) ** 2) ** 0.5
        if is_spatially_aligned(car_box, plate_box) and distance < smallest_distance:
            smallest_distance = distance
            best_match_car_id = car_id
    return best_match_car_id

def listen_for_confirmations(context, tracker):
    pull_confirm = context.socket(zmq.PULL)
    pull_confirm.bind("tcp://*:5557")
    while True:
        data = pull_confirm.recv()
        msg = pickle.loads(data)
        car_id = msg["car_id"]
        plate = msg["confirmed_plate"]
        if car_id in tracker.objects:
            tracker.objects[car_id]["plate_number"] = plate
            tracker.objects[car_id]["confirmed"] = True
            direction = tracker.objects[car_id].get("direction", "Unknown")
            print(f"[EDGE] ✅ Confirmed car {car_id}: Plate '{plate}' | Direction: {direction}")

def main():
    context = zmq.Context()
    pull_socket = context.socket(zmq.PULL)
    pull_socket.bind("tcp://*:5555")

    push_socket = context.socket(zmq.PUSH)
    push_socket.connect("tcp://cloud:5556")

    tracker = CentroidTracker()
    threading.Thread(target=listen_for_confirmations, args=(context, tracker), daemon=True).start()

    car_model = YOLO("/models/yolo11n.pt")
    plate_model = YOLO("/models/license_plate_detector.pt")
    roi_data = load_roi_regions()

    print("[EDGE] Service started.")

    while True:
        data = pull_socket.recv()
        message = pickle.loads(data)
        frame = decode_frame(message["frame"])

        car_results = car_model(frame)[0]
        car_detections = []
        for box in car_results.boxes:
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            if conf > 0.4 and class_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_detections.append((x1, y1, x2, y2))

        tracked_cars = tracker.update(car_detections)

        for car_id, details in tracked_cars.items():
            direction = get_direction_for_bbox(details["bbox"], roi_data)
            details["direction"] = direction

        plate_results = plate_model(frame)[0]
        for box in plate_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            if conf > 0.5 and class_id == 0:
                plate_box = BoundingBox(x1, y1, x2 - x1, y2 - y1, conf)
                car_id = match_plate_to_car(plate_box, tracked_cars)
                if car_id is not None:
                    car_data = tracked_cars[car_id]
                    if not car_data["confirmed"]:
                        cropped = frame[y1:y2, x1:x2]
                        _, buf = cv2.imencode('.jpg', cropped)
                        encoded_crop = base64.b64encode(buf).decode("utf-8")
                        push_socket.send(pickle.dumps({
                            "car_id": car_id,
                            "plate_crop": encoded_crop
                        }))

if __name__ == "__main__":
    main()
