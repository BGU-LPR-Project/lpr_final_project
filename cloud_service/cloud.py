import cv2
import numpy as np
import paddleocr
import util
from formats import process_plate

class CloudService:
    def __init__(self):
        self.reader = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
        self.active = False

    def off(self):
        self.active = False

    def on(self, device='CPU'):
        self.active = True

    def predict(self, plate_img, CB):
        if not self.active:
            CB(None)
            return

        predictions = []
        confidences = []
        for _ in range(3):  # clearly OCR 3 times
            text, conf = self.read_text_from_plate(plate_img)
            if text:
                predictions.append(text)
                confidences.append(conf)

        final_plate = util.weighted_majority_vote(predictions)

        CB(final_plate)

    def read_text_from_plate(self, cropped_plate, confidence_threshold=0.8):
        resized = util.resize_plate(cropped_plate)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        sharp = util.sharpenHBF(blur)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        try:
            results = self.reader.ocr(sharp)

            if not results or results[0] is None:
                return None, 0.0

            high_confidence_results = [
                (line[1][0], line[1][1]) for line in results[0] if line[1][1] >= confidence_threshold
            ]

            if high_confidence_results:
                selected_result, selected_confidence = max(high_confidence_results, key=lambda x: x[1])
                processed_text = process_plate(selected_result)
                return processed_text, selected_confidence

            return None, 0.0

        except Exception as e:
            print(f"OCR error: {e}")
            return None, 0.0
