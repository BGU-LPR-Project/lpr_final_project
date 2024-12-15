from typing import Optional
import cv2
import numpy as np
from video_handler import VideoHandler
from AI1 import *
from AI2 import *

class LPRPipeline:
    def __init__(self, video_path: str):
        """
        Initialize the pipeline with a video file path.
        """
        self.video_processor = VideoHandler(video_path)
        self.motion_detector = MotionDetector()

        # Initialize car detector
        self.car_detector = CarDetector("yolo11n.pt")

        # Initilize license plate detector
        self.license_plate_detector = LicensePlateDetector("license_plate_detector.pt")



    def run(self) -> None:
        """
        Run the license plate recognition pipeline on the video.
        """
        try:
            # Load video
            self.video_processor.load_video()

            while True:
                # Decode a frame
                frame: Optional[np.ndarray] = self.video_processor.decode_frame()
                if frame is None:
                    break

                # Resize the frame
                frame = cv2.resize(frame, (800, 600))

                # Step 1: Detect motion
                motion_boxes = self.motion_detector.detect_motion(frame)

                # Step 2: Car detection with YOLO
                detected_cars = self.car_detector.detect_moving_cars(frame, motion_boxes)

                # Step 3 + 4: Detect license plate and perform ocr
                self.license_plate_detector.detect_license_plates(frame, detected_cars)

                # Show the frame
                cv2.imshow("Moving cars Detection", frame)

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.video_processor.release_resources()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "recordings\\motion5.mov"
    pipeline = LPRPipeline(video_path)
    pipeline.run()
