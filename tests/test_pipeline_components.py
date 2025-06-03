import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from edge_service.edge import EdgeService
from cloud_service.formats import process_plate
from edge_service.bounding_box import BoundingBox

class TestPipelineComponents(unittest.TestCase):
    @patch('edge_service.edge.YOLO')
    def setUp(self, mock_yolo):
        """Set up test fixtures before each test method."""
        # Create a mock frame for testing
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a mock license plate region
        cv2.rectangle(self.test_frame, (100, 100), (200, 150), (255, 255, 255), -1)

        # Mock YOLO model
        mock_yolo_instance = Mock()
        mock_yolo_instance.return_value = Mock()
        mock_yolo_instance.return_value.boxes = Mock()
        mock_yolo_instance.return_value.boxes.conf = [np.array([0.9])]
        mock_yolo_instance.return_value.boxes.cls = [np.array([0])]
        mock_yolo_instance.return_value.boxes.xyxy = [np.array([100, 100, 200, 150])]
        mock_yolo.return_value = mock_yolo_instance

        # Initialize services with mocked model paths
        self.edge_service = EdgeService(
            car_model_path="models/car_model.pt",
            plate_model_path="models/plate_model.pt"
        )

        # Mock the tracker's objects dictionary
        self.edge_service.tracker.objects = {
            1: {"plate_number": "43788503", "confidence": 0.9, "occurs": 2, "done": False, "centroid": (100, 100), "bbox": (100, 100, 200, 200)},
            2: {"plate_number": "80304001", "confidence": 0.8, "occurs": 1, "done": False, "centroid": (300, 100), "bbox": (300, 100, 400, 200)}
        }

        # Initialize the tracker's disappeared dictionary
        self.edge_service.tracker.disappeared = {1: 0, 2: 0}

        # Mock OCR service
        self.ocr_service = Mock()
        self.ocr_service.process_image.return_value = ("43788503", 0.95)  # Return tuple of (text, confidence)

    def test_motion_detection(self):
        """Test motion detection functionality."""
        # Create two frames with motion
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        cv2.rectangle(frame2, (100, 100), (200, 150), (255, 255, 255), -1)

        # Mock the motion detector to return expected results
        self.edge_service.motion_detector.detect_motion = Mock(return_value=[BoundingBox(100, 100, 100, 50)])

        # Test motion detection
        motion_boxes = self.edge_service.motion_detector.detect_motion(frame2)
        self.assertGreater(len(motion_boxes), 0)

        # Test no motion
        self.edge_service.motion_detector.detect_motion = Mock(return_value=[])
        motion_boxes = self.edge_service.motion_detector.detect_motion(frame1)
        self.assertEqual(len(motion_boxes), 0)

    def test_car_detection(self):
        """Test car detection functionality."""
        # Create a frame with a car
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 200), (255, 255, 255), -1)  # Mock car
        motion_boxes = [BoundingBox(100, 100, 200, 100)]  # Mock motion box

        # Create mock box object
        class BoxMock(Mock):
            def __init__(self, x1, y1, x2, y2, conf, cls):
                super().__init__()
                self.conf = [np.array([conf])]
                self.cls = [np.array([cls])]
                self.xyxy = [np.array([x1, y1, x2, y2])]

        # Create mock boxes that are iterable
        class BoxesMock(Mock):
            def __iter__(self):
                return iter([BoxMock(100, 100, 300, 200, 0.9, 2)])

        # Mock the car model to return expected results
        mock_result = Mock()
        mock_result.boxes = BoxesMock()

        # Create a custom mock that is both iterable and subscriptable
        class ModelMock(Mock):
            def __iter__(self):
                return iter([mock_result])

            def __getitem__(self, key):
                return mock_result

        self.edge_service.car_model = Mock(return_value=ModelMock())

        # Test car detection
        cars = self.edge_service.detect_moving_cars(frame, motion_boxes)
        self.assertGreater(len(cars), 0)

    def test_plate_detection(self):
        """Test license plate detection functionality."""
        # Create a frame with a license plate
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 150), (255, 255, 255), -1)  # Mock plate
        detected_cars = {
            1: {"plate_number": "43788503", "confidence": 0.9, "occurs": 2, "done": False, "centroid": (150, 150), "bbox": (100, 100, 200, 200)}
        }

        # Create mock box object
        class BoxMock(Mock):
            def __init__(self, x1, y1, x2, y2, conf, cls):
                super().__init__()
                self.conf = [np.array([conf])]
                self.cls = [np.array([cls])]
                self.xyxy = [np.array([x1, y1, x2, y2])]

        # Create mock boxes that are iterable
        class BoxesMock(Mock):
            def __iter__(self):
                return iter([BoxMock(100, 100, 200, 150, 0.9, 0)])

        # Mock the plate model to return expected results
        mock_result = Mock()
        mock_result.boxes = BoxesMock()

        # Create a custom mock that is both iterable and subscriptable
        class ModelMock(Mock):
            def __iter__(self):
                return iter([mock_result])

            def __getitem__(self, key):
                return mock_result

        self.edge_service.plate_model = Mock(return_value=ModelMock())

        # Mock the region adjuster to return the same coordinates
        self.edge_service.region_adjuster.adjust_region = Mock(return_value=(100, 100, 100, 50))

        # Mock the match_plate_to_car method to return the correct car ID
        self.edge_service.match_plate_to_car = Mock(return_value=1)

        # Mock the is_spatially_aligned method to return True
        self.edge_service.is_spatially_aligned = Mock(return_value=True)

        # Test plate detection
        plates = self.edge_service.detect_license_plate_boxes(frame, detected_cars)
        self.assertGreater(len(plates), 0)
        self.assertEqual(len(plates[1]), 4)  # x, y, w, h

    def test_plate_processing(self):
        """Test plate number processing and formatting."""
        # Test valid plate numbers
        self.assertEqual(process_plate("43788503"), "43788503")
        self.assertEqual(process_plate("43788503 "), "43788503")  # Remove spaces
        self.assertEqual(process_plate("43788503\n"), "43788503")  # Remove newlines

        # Test invalid plate numbers
        self.assertIsNone(process_plate(""))
        self.assertIsNone(process_plate("---"))

        # Test None input
        with self.assertRaises(TypeError):
            process_plate(None)

    def test_tracking_consistency(self):
        """Test vehicle tracking consistency."""
        # Test tracking update
        self.edge_service.update_tracked_vehicle(1, "43788503", 0.95)
        self.assertEqual(self.edge_service.tracker.objects[1]["plate_number"], "43788503")
        self.assertEqual(self.edge_service.tracker.objects[1]["confidence"], 0.95)
        self.assertEqual(self.edge_service.tracker.objects[1]["occurs"], 3)

        # Test new vehicle tracking
        # First add the new vehicle to the tracker
        self.edge_service.tracker.objects[3] = {
            "plate_number": None,
            "confidence": 0.0,
            "occurs": 0,
            "done": False,
            "centroid": (0, 0),
            "bbox": (0, 0, 0, 0)
        }
        self.edge_service.update_tracked_vehicle(3, "12345678", 0.85)
        self.assertEqual(self.edge_service.tracker.objects[3]["plate_number"], "12345678")
        self.assertEqual(self.edge_service.tracker.objects[3]["confidence"], 0.85)
        self.assertEqual(self.edge_service.tracker.objects[3]["occurs"], 0)

    def test_error_handling(self):
        """Test error handling in pipeline components."""
        # Test invalid frame
        with self.assertRaises(ValueError):
            self.edge_service.motion_detector.detect_motion = Mock(side_effect=ValueError("Invalid frame"))
            self.edge_service.motion_detector.detect_motion(None)

        # Test invalid plate image
        with self.assertRaises(ValueError):
            self.ocr_service.process_image = Mock(side_effect=ValueError("Invalid image"))
            self.ocr_service.process_image(None)

        # Test invalid tracking data
        with self.assertRaises(KeyError):
            self.edge_service.update_tracked_vehicle(999, "43788503", 0.9)

if __name__ == '__main__':
    unittest.main()
