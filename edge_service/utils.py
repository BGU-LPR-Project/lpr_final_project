from bounding_box import BoundingBox
from typing import List

def merge_boxes(boxes: List[BoundingBox]) -> List[BoundingBox]:
    """
    Merge overlapping or close bounding boxes into combined boxes.

    Args:
        boxes (List[BoundingBox]): List of bounding boxes to merge.

    Returns:
        List[BoundingBox]: List of merged bounding boxes.
    """
    if not boxes:
        return []

    # Start with a shallow copy of the input boxes
    merged = boxes[:]
    changed = True

    # Repeat merging process until no more merges occur
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)  # Track which boxes have been merged/used

        # Iterate through all boxes
        for i in range(len(merged)):
            if used[i]:
                continue  # Skip if this box was already merged

            box1 = merged[i]

            # Try to merge box1 with any subsequent boxes not yet merged
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                box2 = merged[j]

                # If boxes qualify to merge, merge box2 into box1 and mark box2 used
                if should_merge(box1, box2):
                    box1 = box1.merge_with(box2)
                    used[j] = True
                    changed = True  # Mark that a merge happened this round

            new_merged.append(box1)  # Add the merged (or original) box
            used[i] = True

        merged = new_merged  # Update the list with merged boxes

    return merged

def should_merge(box1: BoundingBox, box2: BoundingBox) -> bool:
    """
    Determine whether two bounding boxes should be merged based on
    spatial proximity, intersection, and IoU (Intersection over Union).

    Args:
        box1 (BoundingBox): First bounding box.
        box2 (BoundingBox): Second bounding box.

    Returns:
        bool: True if boxes should be merged, False otherwise.
    """
    # Calculate a threshold based on a quarter of the smallest box dimension
    threshold = min(box1.width, box1.height, box2.width, box2.height) / 4

    # Check if centers of boxes are close in horizontal and vertical directions
    close_in_x = abs((box1.x + box1.width / 2) - (box2.x + box2.width / 2)) < threshold
    close_in_y = abs((box1.y + box1.height / 2) - (box2.y + box2.height / 2)) < threshold

    # Decide to merge if:
    # - Boxes intersect, OR
    # - Boxes are close and have IoU less than 0.7 (to avoid heavy overlap),
    # OR - One box is fully inside the other
    return (box1.intersects_with(box2) or
            (close_in_x and close_in_y and intersect_over_union(box1, box2) < 0.7) or
            box1.is_inside(box2) or
            box2.is_inside(box1))

def intersect_over_union(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    IoU is the ratio of the area of the intersection to the area of the union
    of the two boxes.

    Args:
        box1 (BoundingBox): First bounding box.
        box2 (BoundingBox): Second bounding box.

    Returns:
        float: IoU value between 0 (no overlap) and 1 (perfect overlap).
    """
    # Calculate intersection rectangle coordinates
    inter_x1 = max(box1.x, box2.x)
    inter_y1 = max(box1.y, box2.y)
    inter_x2 = min(box1.x + box1.width, box2.x + box2.width)
    inter_y2 = min(box1.y + box1.height, box2.y + box2.height)

    # Calculate intersection area (0 if no overlap)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate individual box areas
    box1_area = box1.width * box1.height
    box2_area = box2.width * box2.height

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Return IoU ratio, handle zero division case
    return inter_area / union_area if union_area > 0 else 0

def get_intersection_area(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Calculate the intersection area of two bounding boxes.

    Args:
        box1 (BoundingBox): First bounding box.
        box2 (BoundingBox): Second bounding box.

    Returns:
        float: Area of the intersection region, or 0 if no intersection.
    """
    # Calculate intersection rectangle coordinates
    inter_x1 = max(box1.x, box2.x)
    inter_y1 = max(box1.y, box2.y)
    inter_x2 = min(box1.x + box1.width, box2.x + box2.width)
    inter_y2 = min(box1.y + box1.height, box2.y + box2.height)

    # Compute intersection area (width * height)
    return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

def motion_box_valid_for_car(car_box: BoundingBox, motion_box: BoundingBox) -> bool:
    """
    Validate if a motion detection box is a plausible match for a car bounding box.

    Criteria:
    - The motion box aspect ratio (width/height) is within a reasonable range.
    - The IoU with the car box is above a minimum threshold (0.2).

    Args:
        car_box (BoundingBox): Bounding box of the car.
        motion_box (BoundingBox): Bounding box from motion detection.

    Returns:
        bool: True if the motion box is valid for the car box, False otherwise.
    """
    # motion_area = motion_box.width * motion_box.height
    # if motion_area < 1200:  # Lowered from 2000
    #     return False

    # Calculate aspect ratio of the motion box (avoid division by zero)
    aspect_ratio = motion_box.width / (motion_box.height + 1e-5)

    # Reject if aspect ratio is too narrow or too wide
    if aspect_ratio < 0.7 or aspect_ratio > 4.0:
        return False

    # Calculate IoU between car and motion boxes
    iou = intersect_over_union(car_box, motion_box)

    # Reject if IoU is too low (insufficient overlap)
    if iou < 0.2:
        return False

    return True
