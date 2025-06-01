import cv2
import numpy as np

class RegionAdjuster:
    def __init__(self, frame_width, frame_height):
        """
        Initialize with frame dimensions and default boundary lines.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Horizontal line separating top and bottom halves
        self.line_points = [(0, frame_height // 2), (frame_width, frame_height // 2)]
        
        # Vertical line separating left and right regions, initially centered
        mid_x = frame_width // 2
        mid_y = frame_height // 2
        self.vertical_line_points = [
            (mid_x, mid_y),       # Top point on horizontal boundary
            (mid_x, frame_height) # Bottom point at frame bottom
        ]
        
        self.dragging_point = None

    def select_boundary(self, event, x, y, flags, param):
        """
        Mouse callback to drag and adjust boundary endpoints.

        Adjusts horizontal or vertical line points based on mouse interaction.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Detect if click near any boundary endpoint
            for i, (px, py) in enumerate(self.line_points + self.vertical_line_points):
                if abs(px - x) < 10 and abs(py - y) < 10:
                    self.dragging_point = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_point is not None:
            if self.dragging_point < 2:  # Moving horizontal boundary points vertically
                px, _ = self.line_points[self.dragging_point]
                self.line_points[self.dragging_point] = (px, max(0, min(y, self.frame_height - 1)))
                # Keep vertical line's top point aligned with updated horizontal line
                self.vertical_line_points[0] = self.get_point_on_line(self.line_points[0], self.line_points[1], self.vertical_line_points[0][0])
            else:  # Moving vertical boundary points horizontally
                index = self.dragging_point - 2
                if index == 0:  # Top vertical point moves along horizontal line
                    x_clamped = max(0, min(x, self.frame_width - 1))
                    self.vertical_line_points[index] = self.get_point_on_line(self.line_points[0], self.line_points[1], x_clamped)
                elif index == 1:  # Bottom vertical point moves along bottom edge
                    px, _ = self.vertical_line_points[index]
                    self.vertical_line_points[index] = (max(0, min(x, self.frame_width - 1)), self.frame_height)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = None

    def get_point_on_line(self, point1, point2, x):
        """
        Calculate y-coordinate on line between point1 and point2 for given x.

        Returns integer coordinate (x, y).
        """
        x1, y1 = point1
        x2, y2 = point2
        if x1 == x2:  # Vertical line special case
            return (x1, y1)
        slope = (y2 - y1) / (x2 - x1)
        y = y1 + slope * (x - x1)
        return (int(x), int(y))

    def draw_overlay(self, frame):
        """
        Draw translucent colored overlays for defined regions on the frame.

        Returns the frame with overlays blended.
        """
        overlay = frame.copy()

        # Top region overlay (red)
        horizontal_pts = np.array([
            [0, 0],
            [self.frame_width, 0],
            self.line_points[1],
            self.line_points[0]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [horizontal_pts], (0, 0, 255))

        # Left region overlay (blue)
        left_pts = np.array([
            self.vertical_line_points[0],
            self.line_points[0],
            [0, self.frame_height],
            self.vertical_line_points[1]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [left_pts], (255, 0, 0))

        # Right region overlay (green)
        right_pts = np.array([
            self.line_points[1],
            self.vertical_line_points[0],
            self.vertical_line_points[1],
            [self.frame_width, self.frame_height]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [right_pts], (0, 255, 0))

        # Blend overlay with original frame
        alpha = 0.2
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def draw_labels(self, frame):
        """
        Draw 'Entrance' and 'Exit' labels in center of left and right regions.
        """
        left_label = "Entrance"
        right_label = "Exit"

        left_center_x = self.vertical_line_points[0][0] // 2
        left_center_y = (self.line_points[0][1] + self.line_points[1][1]) // 2

        right_center_x = (self.vertical_line_points[0][0] + self.frame_width) // 2
        right_center_y = (self.line_points[0][1] + self.line_points[1][1]) // 2

        cv2.putText(frame, left_label, (left_center_x, left_center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, right_label, (right_center_x, right_center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_boundary(self, frame):
        """
        Draw boundary lines and endpoint circles on the frame.
        """
        for point in self.line_points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.line(frame, self.line_points[0], self.line_points[1], (0, 255, 0), 2)

        for point in self.vertical_line_points:
            cv2.circle(frame, point, 5, (255, 255, 0), -1)
        cv2.line(frame, self.vertical_line_points[0], self.vertical_line_points[1], (255, 255, 0), 2)

    def is_in_entrance_or_exit(self, bounding_box):
        """
        Classify bounding box as 'Entrance' or 'Exit' based on horizontal position.

        Args:
            bounding_box (tuple): (x, y, width, height)

        Returns:
            str: "Entrance" if left side, else "Exit"
        """
        box_center_x = (bounding_box[0] + bounding_box[2]) // 2
        if box_center_x < self.vertical_line_points[0][0]:
            return "Entrance"
        else:
            return "Exit"

    def apply_roi_mask(self, frame):
        """
        Apply a mask to blank out areas outside the region of interest (ROI).

        Returns the masked frame.
        """
        mask = np.zeros_like(frame, dtype=np.uint8)

        roi_polygon = np.array([
            self.line_points[0],
            self.line_points[1],
            [self.frame_width, self.frame_height],
            [0, self.frame_height]
        ], dtype=np.int32)

        cv2.fillPoly(mask, [roi_polygon], (255, 255, 255))
        roi_frame = cv2.bitwise_and(frame, mask)
        return roi_frame
