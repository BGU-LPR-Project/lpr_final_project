import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
from datetime import datetime
import threading

class CentroidTracker:
    def __init__(self, max_disappeared=5):
        self.next_object_id = 0
        self.objects = OrderedDict()  # Stores object details (centroid, bounding box, plate number)
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.tracked_plates = OrderedDict()
        self.lock = threading.RLock()  

    def register(self, centroid, bbox):
        """Assign a new ID to a detected object."""
        with self.lock:
            self.objects[self.next_object_id] = {
                "centroid": centroid,
                "bbox": bbox,
                "plate_number": str(),
                "plate_box": None,
                "direction": None,
                "confidence": 0.0,
                "last_timestamp": datetime.now(),
                "occurs": 0,
                "done": False,
            }
            self.disappeared[self.next_object_id] = 0
            self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object that is no longer detected."""
        with self.lock:
            del self.objects[object_id]
            del self.disappeared[object_id]

    def update(self, detections, max_distance=200):
        """Update object tracking based on new detections and a max distance threshold."""
        with self.lock:
            if len(detections) == 0:
                for object_id in list(self.disappeared.keys()):
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
                return self.objects

            # Compute centroids of new detections
            input_centroids = np.zeros((len(detections), 2), dtype="int")
            for i, (start_x, start_y, end_x, end_y) in enumerate(detections):
                c_x = int((start_x + end_x) / 2.0)
                c_y = int((start_y + end_y) / 2.0)
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
                    if D[row, col] > max_distance:
                        continue  # Ignore matches beyond the allowed threshold

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


    def non_max_suppression_fast(self, boxes, overlap_thresh=0.5):
        with self.lock:
            if len(boxes) == 0:
                return []
            boxes = np.array(boxes)
            pick = []

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(y2)

            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                overlap = (w * h) / area[idxs[:last]]
                idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

            return boxes[pick].astype("int")
    
    def update_tracked_plate(self, object_id, plate_number):
        with self.lock:
            self.tracked_plates[object_id] = plate_number