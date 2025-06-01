import cv2
import numpy as np
import paddleocr
import util
from formats import process_plate
import os

class CloudService:
    """
    Cloud-based OCR service using PaddleOCR for license plate recognition.
    Provides activation control and prediction interface.
    """
    
    def __init__(self):
        """
        Initialize the PaddleOCR model with predefined configurations and model paths.
        """
        self.active = False
        self.reader = paddleocr.PaddleOCR(
            use_angle_cls=True,
            lang='en',
            det_model_dir='/app/models/paddle/det',
            rec_model_dir='/app/models/paddle/rec',
            cls_model_dir='/app/models/paddle/cls',
            use_gpu=False
        )

    def off(self):
        """
        Deactivate the OCR service.
        """
        self.active = False

    def on(self, device='CPU'):
        """
        Activate the OCR service.
        """
        self.active = True

    def predict(self, plate_img, CB):
        """
        Predict text from a license plate image and return result via callback.
        
        Args:
            plate_img (np.ndarray): Cropped license plate image.
            CB (callable): Callback function to handle (plate_text, confidence).
        """
        if not self.active:
            CB((str(), 0.0))  # Return empty result if service is inactive
            return

        text, conf = self.read_text_from_plate(plate_img)
        print(text)

        processed_plate = process_plate(text) if text else None
        print(processed_plate)

        if not processed_plate:
            CB((str(), 0.0))  # Return empty result if plate format is invalid
            return

        CB((processed_plate, conf))  # Return final prediction

    def read_text_from_plate(self, cropped_plate, confidence_threshold=0.8):
        """
        Preprocess license plate image and extract text using OCR.

        Args:
            cropped_plate (np.ndarray): Input license plate image.
            confidence_threshold (float): Minimum confidence to accept OCR text.

        Returns:
            Tuple[str, float]: Extracted text and associated confidence score.
        """
        # Resize to standard size
        resized = util.resize_plate(cropped_plate)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Denoise with Gaussian blur
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Sharpen image using High Boost Filter
        sharp = util.sharpenHBF(blur)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        ocr_ready = sharp

        # Save image for inspection/debugging
        filename = os.path.join("/app/models", "plate-processed.jpg")
        cv2.imwrite(filename, ocr_ready)

        # Perform OCR
        try:
            results = self.reader.ocr(ocr_ready)

            selected_result = None
            selected_confidence = 0.0

            # Return early if OCR failed
            if not results or results[0] is None:
                return None, 0.0

            # Filter results above confidence threshold
            high_confidence_results = [
                (line[1][0], line[1][1]) for line in results[0] if line[1][1] >= confidence_threshold
            ]

            if high_confidence_results:
                # Choose the result with the highest confidence
                selected_result, selected_confidence = max(high_confidence_results, key=lambda x: x[1])

            return selected_result, selected_confidence

        except Exception as e:
            print(f"OCR error: {e}")
            return None, 0.0
