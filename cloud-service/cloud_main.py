import cv2
import zmq
import base64
import pickle
import numpy as np
from paddleocr import PaddleOCR
from collections import defaultdict


# ---------- Preprocessing ----------
def resize_plate(plate: np.ndarray) -> np.ndarray:
    return cv2.resize(plate, (300, 75))

def sharpenHBF(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def decode_crop(encoded_str):
    buffer = base64.b64decode(encoded_str)
    np_arr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def preprocess_for_ocr(plate: np.ndarray) -> np.ndarray:
    resized = resize_plate(plate)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = sharpenHBF(blur)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------- Main ----------
def main():
    context = zmq.Context()
    pull_socket = context.socket(zmq.PULL)
    pull_socket.bind("tcp://*:5556")  # from edge

    # NEW: PUSH socket back to edge
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect("tcp://edge:5557")

    # Preload OCR model
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
    _ = ocr_engine.ocr(np.ones((100, 300, 3), dtype=np.uint8))  # dummy call to preload

    print("[CLOUD] OCR service running...")

    seen_counts = defaultdict(lambda: defaultdict(int))  # car_id -> plate_text -> count

    while True:
        data = pull_socket.recv()
        message = pickle.loads(data)

        car_id = message["car_id"]
        plate_crop = decode_crop(message["plate_crop"])
        prepped = preprocess_for_ocr(plate_crop)

        try:
            results = ocr_engine.ocr(prepped)
            if results and results[0]:
                best = max(results[0], key=lambda line: line[1][1])
                plate_text = best[1][0]
                confidence = best[1][1]

                print(f"[CLOUD] car_id={car_id} Plate: {plate_text} | Confidence: {confidence:.2f}")
                if confidence > 0.75:
                    seen_counts[car_id][plate_text] += 1

                if seen_counts[car_id][plate_text] == 3:
                    print(f"[CLOUD] ✅ Confirmed plate '{plate_text}' for car {car_id}")
                    push_socket.send(pickle.dumps({
                        "car_id": car_id,
                        "confirmed_plate": plate_text,
                        "confidence": confidence
                    }))

        except Exception as e:
            print(f"[CLOUD] OCR error: {e}")

if __name__ == "__main__":
    main()
