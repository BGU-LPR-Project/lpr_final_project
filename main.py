from typing import Optional
import cv2
import numpy as np
from video_handler import VideoHandler
from AI1 import *
from AI2 import *

class LPRPipeline:
    def __init__(self, video_path: str):
        self.video_processor = VideoHandler(video_path)
        self.motion_detector = MotionDetector()  # Make sure it uses the updated MotionDetector
        self.car_detector = CarDetector("yolo11n.pt")
        self.license_plate_detector = LicensePlateDetector("license_plate_detector.pt")

    def run(self):
        try:
            self.video_processor.load_video()
            while True:
                frame = self.video_processor.decode_frame(skip_frames=2)
                if frame is None:
                    break
                frame = cv2.resize(frame, (800, 600))
                motion_boxes = self.motion_detector.detect_motion(frame)
                detected_cars = self.car_detector.detect_moving_cars(frame, motion_boxes)
                self.license_plate_detector.detect_license_plates(frame, detected_cars)
                cv2.imshow("Moving cars Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.video_processor.release_resources()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "recordings\\videoplayback.mp4"
    pipeline = LPRPipeline(video_path)
    pipeline.run()
