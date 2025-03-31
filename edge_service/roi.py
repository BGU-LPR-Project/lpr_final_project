import cv2
import numpy as np

class RegionAdjuster:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Line separating top and bottom
        self.line_points = [(0, frame_height // 2), (frame_width, frame_height // 2)]
        
        # Vertical line separating left and right, initialized in the middle
        mid_x = frame_width // 2
        mid_y = frame_height // 2
        self.vertical_line_points = [
            (mid_x, mid_y),  # Top point starts in the middle of the boundary
            (mid_x, frame_height)  # Bottom point starts at the middle bottom
        ]
        
        self.dragging_point = None

    def select_boundary(self, event, x, y, flags, param):
        """Mouse callback to adjust boundary endpoints."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the user clicked near one of the points
            for i, (px, py) in enumerate(self.line_points + self.vertical_line_points):
                if abs(px - x) < 10 and abs(py - y) < 10:
                    self.dragging_point = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_point is not None:
            if self.dragging_point < 2:  # Adjust horizontal line points
                px, _ = self.line_points[self.dragging_point]
                self.line_points[self.dragging_point] = (px, max(0, min(y, self.frame_height - 1)))
                # Automatically adjust the vertical line's top point to align with the new boundary
                self.vertical_line_points[0] = self.get_point_on_line(self.line_points[0], self.line_points[1], self.vertical_line_points[0][0])
            else:  # Adjust vertical line points
                index = self.dragging_point - 2
                if index == 0:  # Top vertical point moves along the first boundary line
                    x_clamped = max(0, min(x, self.frame_width - 1))
                    self.vertical_line_points[index] = self.get_point_on_line(self.line_points[0], self.line_points[1], x_clamped)
                elif index == 1:  # Bottom vertical point moves along the bottom
                    px, _ = self.vertical_line_points[index]
                    self.vertical_line_points[index] = (max(0, min(x, self.frame_width - 1)), self.frame_height)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = None

    def get_point_on_line(self, point1, point2, x):
        """Calculate the point (x, y) on a line given x."""
        x1, y1 = point1
        x2, y2 = point2
        if x1 == x2:  # Vertical line, y is constant
            return (x1, y1)
        slope = (y2 - y1) / (x2 - x1)
        y = y1 + slope * (x - x1)
        return (int(x), int(y))

    def draw_overlay(self, frame):
        """Draw transparent overlays for the regions and the boundaries."""
        overlay = frame.copy()

        # Overlay for the horizontal boundary
        horizontal_pts = np.array([
            [0, 0],  # Top-left corner
            [self.frame_width, 0],  # Top-right corner
            self.line_points[1],  # Right endpoint of the horizontal line
            self.line_points[0]   # Left endpoint of the horizontal line
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [horizontal_pts], (0, 0, 255))  # Red color in BGR

        # Overlay for the left side (blue)
        left_pts = np.array([
            self.vertical_line_points[0],
            self.line_points[0],
            [0, self.frame_height],
            self.vertical_line_points[1]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [left_pts], (255, 0, 0))  # Blue color in BGR

        # Overlay for the right side (green)
        right_pts = np.array([
            self.line_points[1],
            self.vertical_line_points[0],
            self.vertical_line_points[1],
            [self.frame_width, self.frame_height]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [right_pts], (0, 255, 0))  # Green color in BGR

        # Blend the overlay with transparency
        alpha = 0.2  # Transparency factor for side overlays
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def draw_labels(self, frame):
        """Draw labels in the middle of the left and right overlay regions."""
        left_label = "Entrance"
        right_label = "Exit"

        # Calculate center points for the left and right regions
        left_center_x = self.vertical_line_points[0][0] // 2
        left_center_y = (self.line_points[0][1] + self.line_points[1][1]) // 2

        right_center_x = (self.vertical_line_points[0][0] + self.frame_width) // 2
        right_center_y = (self.line_points[0][1] + self.line_points[1][1]) // 2

        # Draw the labels
        cv2.putText(frame, left_label, (left_center_x, left_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, right_label, (right_center_x, right_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    def draw_boundary(self, frame):
        """Draw the boundaries and their endpoints on the frame."""
        for point in self.line_points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Draw horizontal line points
        cv2.line(frame, self.line_points[0], self.line_points[1], (0, 255, 0), 2)  # Draw horizontal line

        for point in self.vertical_line_points:
            cv2.circle(frame, point, 5, (255, 255, 0), -1)  # Draw vertical line points
        cv2.line(frame, self.vertical_line_points[0], self.vertical_line_points[1], (255, 255, 0), 2)  # Draw vertical line


    def is_in_entrance_or_exit(self, bounding_box):
        """Determine if a bounding box is in the 'Entrance' or 'Exit' region."""
        box_center_x = (bounding_box[0] + bounding_box[2]) // 2
        if box_center_x < self.vertical_line_points[0][0]:
            return "Entrance"
        else:
            return "Exit"


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
