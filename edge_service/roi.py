import cv2
import numpy as np

class RegionAdjuster:
    def __init__(self):
        """
        RegionAdjuster with no initial region config.
        Must be populated via `load_region_config(region_data, config_frame_size)`.
        """
        self.configured = False
        self.line_points = None
        self.vertical_line_points = None
        self.entrance_side = None
        self.scale_x = 1.0
        self.scale_y = 1.0

    def load_region_config(self, region_data, config_frame_size=(960,540)):
        """
        Load region coordinates and scale them to match actual frame size.

        Args:
            region_data (dict): Contains 'horizontal', 'vertical', and 'entranceSide'.
            config_frame_size (tuple): The width and height the region was drawn on (e.g., canvas size).
        """
        config_w, config_h = config_frame_size
        self.entrance_side = region_data.get('entranceSide', 'left')
        self.original_config_size = config_frame_size

        # Prepare for scaling later
        self.line_points = [(pt['x'], pt['y']) for pt in region_data['horizontal']]
        self.vertical_line_points = [(pt['x'], pt['y']) for pt in region_data['vertical']]

        self.configured = True

    def _scale_points(self, points, frame_shape):
        """
        Scale region config points to actual frame dimensions.
        """
        frame_h, frame_w = frame_shape[:2]
        config_w, config_h = self.original_config_size
        scale_x = frame_w / config_w
        scale_y = frame_h / config_h
        return [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in points]

    def apply_roi_mask(self, frame):
        """
        Apply ROI mask below the horizontal line. If not configured, return original.
        """
        if not self.configured:
            return frame

        line_pts = self._scale_points(self.line_points, frame.shape)
        mask = np.zeros_like(frame, dtype=np.uint8)

        roi_polygon = np.array([
            line_pts[0],
            line_pts[1],
            (frame.shape[1], frame.shape[0]),
            (0, frame.shape[0])
        ], dtype=np.int32)

        cv2.fillPoly(mask, [roi_polygon], (255, 255, 255))
        return cv2.bitwise_and(frame, mask)

    def is_in_entrance_or_exit(self, bounding_box, frame):
        """
        Determine whether the object is in 'Entrance' or 'Exit' region.
        If unconfigured, return 'N/A'.

        Args:
            bounding_box (tuple): (x1, y1, x2, y2)
            frame (np.ndarray): Used for scaling.
        Returns:
            str: "Entrance", "Exit", or "N/A"
        """
        if not self.configured:
            return "N/A"

        vertical_pts = self._scale_points(self.vertical_line_points, frame.shape)
        vert_x = vertical_pts[0][0]  # vertical boundary x

        x1, y1, x2, y2 = bounding_box
        center_x = (x1 + x2) // 2

        if self.entrance_side == 'left':
            return "Entrance" if center_x < vert_x else "Exit"
        else:
            return "Entrance" if center_x > vert_x else "Exit"

    def unset_region_config(self):
        """
        Unsets the currently configured region by marking the configuration as inactive.

        This method resets the `configured` flag to False, effectively disabling any
        previously set region boundaries or logic that depends on region configuration.
        """
        self.configured = False
