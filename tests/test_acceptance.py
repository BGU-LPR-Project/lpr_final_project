import sys
import os
import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from edge_service.edge import EdgeService
from cloud_service.cloud import CloudService
import time

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestLPRSystemAcceptance(unittest.TestCase):
    @patch('edge_service.edge.YOLO')
    def setUp(self, mock_yolo):
        """Set up test fixtures before each test method."""
        # Initialize services
        self.edge_service = EdgeService(
            car_model_path="models/car_model.pt",
            plate_model_path="models/plate_model.pt"
        )
        self.cloud_service = CloudService()

        # Create a mock frame for testing
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_single_vehicle_detection_and_recognition(self):
        """Test the complete pipeline for a single vehicle."""
        # Create a frame with a car and license plate
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a car
        cv2.rectangle(frame, (100, 100), (300, 200), (255, 255, 255), -1)
        # Draw a license plate
        cv2.rectangle(frame, (150, 150), (250, 170), (0, 0, 255), -1)

        # Mock the YOLO models
        self.edge_service.car_model = Mock(return_value=self._create_mock_yolo_result(100, 100, 300, 200, 0.9, 2))
        self.edge_service.plate_model = Mock(return_value=self._create_mock_yolo_result(150, 150, 250, 170, 0.9, 0))

        # Mock the OCR service
        self.cloud_service.ocr = Mock()
        self.cloud_service.ocr.process_image.return_value = ("ABC123", 0.95)

        # Process the frame
        result = {}
        self.edge_service.predict(frame, lambda x: result.update(x))

        # Verify the results
        self.assertGreater(len(result), 0)
        self.assertIn("plate_number", result[1])
        self.assertEqual(result[1]["plate_number"], "ABC123")
        self.assertGreater(result[1]["confidence"], 0.8)

    def test_multiple_vehicles_detection(self):
        """Test detection and tracking of multiple vehicles."""
        # Create frames with multiple vehicles
        frames = []
        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw cars at different positions
            cv2.rectangle(frame, (100 + i*50, 100), (300 + i*50, 200), (255, 255, 255), -1)
            frames.append(frame)

        # Mock the YOLO models
        self.edge_service.car_model = Mock(return_value=self._create_mock_yolo_result(100, 100, 300, 200, 0.9, 2))
        self.edge_service.plate_model = Mock(return_value=self._create_mock_yolo_result(150, 150, 250, 170, 0.9, 0))

        # Process each frame
        results = []
        for frame in frames:
            result = {}
            self.edge_service.predict(frame, lambda x: result.update(x))
            results.append(result)

        # Verify tracking consistency
        self.assertGreater(len(results[-1]), 0)
        # Check that vehicles are tracked consistently across frames
        for i in range(1, len(results)):
            self.assertEqual(set(results[i].keys()), set(results[i-1].keys()))

    def test_low_light_conditions(self):
        """Test system performance in low light conditions."""
        # Create a dark frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a car with low contrast
        cv2.rectangle(frame, (100, 100), (300, 200), (50, 50, 50), -1)
        # Draw a license plate with low contrast
        cv2.rectangle(frame, (150, 150), (250, 170), (30, 30, 30), -1)

        # Mock the YOLO models with lower confidence
        self.edge_service.car_model = Mock(return_value=self._create_mock_yolo_result(100, 100, 300, 200, 0.6, 2))
        self.edge_service.plate_model = Mock(return_value=self._create_mock_yolo_result(150, 150, 250, 170, 0.6, 0))

        # Mock the OCR service with lower confidence
        self.cloud_service.ocr = Mock()
        self.cloud_service.ocr.process_image.return_value = ("ABC123", 0.7)

        # Process the frame
        result = {}
        self.edge_service.predict(frame, lambda x: result.update(x))

        # Verify the results
        self.assertGreater(len(result), 0)
        self.assertIn("plate_number", result[1])
        self.assertEqual(result[1]["plate_number"], "ABC123")
        self.assertLess(result[1]["confidence"], 0.8)

    def test_system_performance(self):
        """Test system performance and response time."""
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 200), (255, 255, 255), -1)

        # Mock the services
        self.edge_service.car_model = Mock(return_value=self._create_mock_yolo_result(100, 100, 300, 200, 0.9, 2))
        self.edge_service.plate_model = Mock(return_value=self._create_mock_yolo_result(150, 150, 250, 170, 0.9, 0))

        # Measure processing time
        start_time = time.time()
        result = {}
        self.edge_service.predict(frame, lambda x: result.update(x))
        processing_time = time.time() - start_time

        # Verify performance requirements
        self.assertLess(processing_time, 1.0)  # Should process within 1 second
        self.assertGreater(len(result), 0)

    def test_authorization_checking(self):
        """Test the system's ability to check vehicle authorization."""
        # Create a frame with a car and license plate
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 200), (255, 255, 255), -1)
        cv2.rectangle(frame, (150, 150), (250, 170), (0, 0, 255), -1)

        # Mock the YOLO models
        self.edge_service.car_model = Mock(return_value=self._create_mock_yolo_result(100, 100, 300, 200, 0.9, 2))
        self.edge_service.plate_model = Mock(return_value=self._create_mock_yolo_result(150, 150, 250, 170, 0.9, 0))

        # Mock the cloud service for authorization
        self.cloud_service.check_authorization = Mock(return_value=True)
        self.cloud_service.ocr = Mock()
        self.cloud_service.ocr.process_image.return_value = ("ABC123", 0.95)

        # Process the frame
        result = {}
        self.edge_service.predict(frame, lambda x: result.update(x))

        # Verify authorization check
        self.assertGreater(len(result), 0)
        self.assertIn("authorized", result[1])
        self.assertTrue(result[1]["authorized"])

    def test_direction_detection(self):
        """Test the system's ability to detect vehicle direction."""
        # Create frames with a moving vehicle
        frames = []
        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw car moving from left to right
            cv2.rectangle(frame, (100 + i*50, 100), (300 + i*50, 200), (255, 255, 255), -1)
            frames.append(frame)

        # Mock the YOLO models
        self.edge_service.car_model = Mock(return_value=self._create_mock_yolo_result(100, 100, 300, 200, 0.9, 2))
        self.edge_service.plate_model = Mock(return_value=self._create_mock_yolo_result(150, 150, 250, 170, 0.9, 0))

        # Process frames
        results = []
        for frame in frames:
            result = {}
            self.edge_service.predict(frame, lambda x: result.update(x))
            results.append(result)

        # Verify direction detection
        self.assertGreater(len(results[-1]), 0)
        self.assertIn("direction", results[-1][1])
        self.assertEqual(results[-1][1]["direction"], "right")

    def test_partial_plate_matching(self):
        """Test the system's ability to handle partial plate matches."""
        # Create a frame with a partially visible license plate
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 200), (255, 255, 255), -1)
        # Draw a partially visible plate
        cv2.rectangle(frame, (150, 150), (200, 170), (0, 0, 255), -1)

        # Mock the YOLO models
        self.edge_service.car_model = Mock(return_value=self._create_mock_yolo_result(100, 100, 300, 200, 0.9, 2))
        self.edge_service.plate_model = Mock(return_value=self._create_mock_yolo_result(150, 150, 200, 170, 0.9, 0))

        # Mock the OCR service to return partial plate
        self.cloud_service.ocr = Mock()
        self.cloud_service.ocr.process_image.return_value = ("ABC12", 0.7)  # Partial plate

        # Process the frame
        result = {}
        self.edge_service.predict(frame, lambda x: result.update(x))

        # Verify partial matching
        self.assertGreater(len(result), 0)
        self.assertIn("plate_number", result[1])
        self.assertIn("confidence", result[1])
        self.assertLess(result[1]["confidence"], 0.8)  # Lower confidence for partial match

    def test_system_reliability(self):
        """Test the system's reliability over multiple frames."""
        # Create a sequence of frames with varying conditions
        frames = []
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Vary the car position and lighting
            brightness = 255 if i % 2 == 0 else 100
            cv2.rectangle(frame, (100 + i*20, 100), (300 + i*20, 200), (brightness, brightness, brightness), -1)
            frames.append(frame)

        # Mock the YOLO models with varying confidence
        def get_mock_result(frame_idx):
            conf = 0.9 if frame_idx % 2 == 0 else 0.7
            return self._create_mock_yolo_result(100 + frame_idx*20, 100, 300 + frame_idx*20, 200, conf, 2)

        self.edge_service.car_model = Mock(side_effect=get_mock_result)
        self.edge_service.plate_model = Mock(return_value=self._create_mock_yolo_result(150, 150, 250, 170, 0.9, 0))

        # Process frames
        results = []
        for frame in frames:
            result = {}
            self.edge_service.predict(frame, lambda x: result.update(x))
            results.append(result)

        # Verify system reliability
        self.assertGreater(len(results[-1]), 0)
        # Check that tracking is maintained despite varying conditions
        for i in range(1, len(results)):
            self.assertEqual(set(results[i].keys()), set(results[i-1].keys()))

    def _create_mock_yolo_result(self, x1, y1, x2, y2, conf, cls):
        """Helper method to create mock YOLO results."""
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.conf = [np.array([conf])]
        mock_result.boxes.cls = [np.array([cls])]
        mock_result.boxes.xyxy = [np.array([x1, y1, x2, y2])]
        return [mock_result]

if __name__ == '__main__':
    unittest.main()
