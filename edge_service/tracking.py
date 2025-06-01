import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
from datetime import datetime
import threading

class CentroidTracker:
    def __init__(self, max_disappeared=5):
        """
        Initialize the centroid tracker.

        Args:
            max_disappeared (int): Maximum number of consecutive frames an object is allowed
                                   to be missing before deregistration.
        """
        self.next_object_id = 0
        # Stores object details keyed by object ID, including centroid, bbox, plate info, etc.
        self.objects = OrderedDict()
        # Tracks how many consecutive frames each object has been missing
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        # Stores plate numbers tracked per object ID
        self.tracked_plates = OrderedDict()
        # Thread lock to ensure thread safety for updates and modifications
        self.lock = threading.RLock()  

    def register(self, centroid, bbox):
        """
        Register a new object with a unique ID.

        Args:
            centroid (tuple): The centroid (x, y) of the detected object.
            bbox (tuple): The bounding box (startX, startY, endX, endY) of the detected object.
        """
        with self.lock:
            self.objects[self.next_object_id] = {
                "centroid": centroid,
                "bbox": bbox,
                "plate_number": str(),  # Empty string initially
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
        """
        Deregister an object that is no longer detected.

        Args:
            object_id (int): The ID of the object to remove.
        """
        with self.lock:
            del self.objects[object_id]
            del self.disappeared[object_id]

    def update(self, detections, max_distance=200):
        """
        Update tracked objects with new detections.

        Args:
            detections (list of tuples): List of bounding boxes for detected objects,
                                         each as (startX, startY, endX, endY).
            max_distance (float): Maximum allowed distance between old and new centroids
                                  to consider it the same object.

        Returns:
            OrderedDict: Updated dictionary of tracked objects with their metadata.
        """
        with self.lock:
            # No detections: mark all existing objects as disappeared
            if len(detections) == 0:
                for object_id in list(self.disappeared.keys()):
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
                return self.objects

            # Compute centroids for all detections
            input_centroids = np.zeros((len(detections), 2), dtype="int")
            for i, (start_x, start_y, end_x, end_y) in enumerate(detections):
                c_x = int((start_x + end_x) / 2.0)
                c_y = int((start_y + end_y) / 2.0)
                input_centroids[i] = (c_x, c_y)

            # If no objects currently tracked, register all detections
            if len(self.objects) == 0:
                for i in range(len(input_centroids)):
                    self.register(input_centroids[i], detections[i])
            else:
                # List of existing object IDs and their centroids
                object_ids = list(self.objects.keys())
                object_centroids = [obj["centroid"] for obj in self.objects.values()]
                # Compute distance matrix between existing and new centroids
                D = dist.cdist(np.array(object_centroids), input_centroids)

                # Sort rows based on minimum distance (closest matches first)
                rows = D.min(axis=1).argsort()
                # Find the column index of minimum distance in each row sorted by rows
                cols = D.argmin(axis=1)[rows]

                used_rows = set()
                used_cols = set()

                # Match objects to detections based on distances and max_distance threshold
                for row, col in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    if D[row, col] > max_distance:
                        continue  # Skip if too far

                    object_id = object_ids[row]
                    # Update tracked object info with new centroid and bbox
                    self.objects[object_id]["centroid"] = input_centroids[col]
                    self.objects[object_id]["bbox"] = detections[col]
                    self.disappeared[object_id] = 0  # Reset disappeared count
                    used_rows.add(row)
                    used_cols.add(col)

                # Find unmatched existing objects and mark them as disappeared
                unused_rows = set(range(D.shape[0])).difference(used_rows)
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

                # Register new objects for unmatched detections
                unused_cols = set(range(D.shape[1])).difference(used_cols)
                for col in unused_cols:
                    self.register(input_centroids[col], detections[col])

            return self.objects

    def non_max_suppression_fast(self, boxes, overlap_thresh=0.5):
        """
        Perform non-maximum suppression to eliminate overlapping bounding boxes.

        Args:
            boxes (list of tuples): List of bounding boxes (startX, startY, endX, endY).
            overlap_thresh (float): Threshold for overlapping areas to suppress boxes.

        Returns:
            list: Bounding boxes after suppression.
        """
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

                # Compute overlap area between the picked box and the rest
                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                overlap = (w * h) / area[idxs[:last]]

                # Delete indices where overlap exceeds threshold
                idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

            return boxes[pick].astype("int")

    def update_tracked_plate(self, object_id, plate_number):
        """
        Update the plate number associated with a tracked object.

        Args:
            object_id (int): The ID of the tracked object.
            plate_number (str): The detected plate number to associate.
        """
        with self.lock:
            self.tracked_plates[object_id] = plate_number