import threading
import queue
import cv2
import numpy as np
from formats import process_plate
from util import resize_plate, sharpenHBF
import paddleocr
import logging
logging.getLogger('ppocr').setLevel(logging.WARNING)


class CloudService:
    def __init__(self):
        self.reader = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
        self.input_queue = queue.Queue()
        self.thread = None
        self.active = False

    def on(self, device='CPU'):
        self.active = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def off(self):
        self.active = False
        if self.thread:
            self.thread.join()

    def predict(self, predictData, CB):
        frame, detections = predictData
        results = {}
        for obj_id, data in detections.items():
            x1, y1, x2, y2 = data['bbox']
            plate_img = frame[y1:y2, x1:x2]
            plate_number = self.ocr_plate(plate_img)
            results[obj_id] = plate_number
            print(f"[CLOUD] ID {obj_id}: {plate_number}")
        if CB:
            CB(results)
        return results

    def _run(self):
        while self.active:
            try:
                item = self.input_queue.get(timeout=1)
                self.predict(item, None)
                self.input_queue.task_done()
            except queue.Empty:
                continue

    def ocr_plate(self, img):
        try:
            if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                print("[CLOUD][ERROR] Empty or invalid plate image.")
                return "ERROR"

            resized = resize_plate(img)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            sharp = sharpenHBF(blur)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)

            cv2.imshow("Cropped Plate", sharp)
            cv2.waitKey(1)

            results = self.reader.ocr(sharp)
            if not results or not results[0]:
                print("[CLOUD][OCR] No text detected.")
                return "UNRECOGNIZED"

            for line in results[0]:
                text, conf = line[1]
                print(f"[CLOUD][OCR] Text: {text}, Conf: {conf:.2f}")
                if conf > 0.8:
                    return process_plate(text)
            return "UNRECOGNIZED"

        except Exception as e:
            print(f"[CLOUD][EXCEPTION] OCR failed: {e}")
            return "ERROR"

