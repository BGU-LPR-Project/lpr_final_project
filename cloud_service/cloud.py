import cv2
import numpy as np
import paddleocr
import util
from formats import process_plate
import os

class CloudService:
    def __init__(self):
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
        self.active = False

    def on(self, device='CPU'):
        self.active = True

    def predict(self, plate_img, CB):
        if not self.active:
            CB((str(), 0.0))
            return

        text, conf = self.read_text_from_plate(plate_img)
        print(text)
        processed_plate = process_plate(text) if text else None
        print(processed_plate)

        if not processed_plate:
            CB((str(), 0.0))
            return

        CB((processed_plate, conf))

    def read_text_from_plate(self, cropped_plate, confidence_threshold=0.8):
         # Resize the plate to standard size
        resized = util.resize_plate(cropped_plate)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Remove noise            
        blur = cv2.GaussianBlur(gray, (3,3), 0)

        # Sharpen using HBF
        sharp = util.sharpenHBF(blur)

        # Clip values to maintain valid range
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        ocr_ready = sharp

        filename = os.path.join("/app/models", "plate-processed.jpg")
        cv2.imwrite(filename, ocr_ready)

        # Use OCR
        try:
            results = self.reader.ocr(ocr_ready)

            selected_result = None
            selected_confidence = 0.0

            # Check if results is None
            if not results or results[0] is None:
                return None, 0.0
            
            # Extract text with high confidence
            high_confidence_results = [
                (line[1][0], line[1][1]) for line in results[0] if line[1][1] >= confidence_threshold
            ]

            if high_confidence_results:
                # Select the result with the highest confidence
                selected_result, selected_confidence = max(high_confidence_results, key=lambda x: x[1])

            return selected_result, selected_confidence

        except Exception as e:
            print(f"OCR error: {e}")
            return None, 0.0
