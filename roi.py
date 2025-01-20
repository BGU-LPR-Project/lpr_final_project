import cv2
import numpy as np

class RegionAdjuster:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.line_points = [(0, frame_height // 2), (frame_width, frame_height // 2)]
        self.dragging_point = None

    def select_boundary(self, event, x, y, flags, param):
        """Mouse callback to adjust boundary endpoints."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the user clicked near one of the points
            for i, (px, py) in enumerate(self.line_points):
                if abs(px - x) < 10 and abs(py - y) < 10:
                    self.dragging_point = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_point is not None:
            # Move the selected point vertically, within frame boundaries
            px, _ = self.line_points[self.dragging_point]
            self.line_points[self.dragging_point] = (px, max(0, min(y, self.frame_height - 1)))
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = None

    def draw_overlay(self, frame):
        """Draw a transparent red overlay above the slanting boundary."""
        overlay = frame.copy()

        # Define the polygon above the line
        pts = np.array([
            [0, 0],  # Top-left corner
            [self.frame_width, 0],  # Top-right corner
            self.line_points[1],  # Right endpoint of the line
            self.line_points[0]   # Left endpoint of the line
        ], dtype=np.int32)

        # Fill the polygon with red
        cv2.fillPoly(overlay, [pts], (0, 0, 255))  # Red color in BGR

        # Blend the overlay with transparency
        alpha = 0.4  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def draw_boundary(self, frame):
        """Draw the boundary and its endpoints on the frame."""
        for point in self.line_points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Draw points
        cv2.line(frame, self.line_points[0], self.line_points[1], (0, 255, 0), 2)  # Draw line

    def apply_roi_mask(self, frame):
        """Blank out the area outside the ROI."""
        mask = np.zeros_like(frame, dtype=np.uint8)

        # Define the polygon for the ROI area
        roi_polygon = np.array([
            self.line_points[0],  # Left endpoint of the line
            self.line_points[1],  # Right endpoint of the line
            [self.frame_width, self.frame_height],  # Bottom-right corner
            [0, self.frame_height]  # Bottom-left corner
        ], dtype=np.int32)

        # Fill the ROI area on the mask
        cv2.fillPoly(mask, [roi_polygon], (255, 255, 255))  # White ROI

        # Apply the mask to the frame
        roi_frame = cv2.bitwise_and(frame, mask)
        return roi_frame
