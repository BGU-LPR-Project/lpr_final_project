import cv2
from video_handler import VideoHandler
from edge import *
from cloud import *
from roi import RegionAdjuster

class LPRPipeline:
    def __init__(self, video_path: str):
        self.video_processor = VideoHandler(video_path)
        self.motion_detector = MotionDetector()  # Make sure it uses the updated MotionDetector
        self.car_detector = CarDetector("yolo11n.pt")
        self.license_plate_detector = LicensePlateDetector("license_plate_detector.pt", [
            r'[A-Z]{2}[0-9]{2}[A-Z]{3}'
        ])

    def run(self):
        # Create a blank frame for demonstration
        frame_width, frame_height = 800, 600

        # Initialize the LineAdjuster
        region_adjuster = RegionAdjuster(frame_width, frame_height)

        cv2.namedWindow("LPR-Control")
        cv2.setMouseCallback("LPR-Control", region_adjuster.select_boundary)

        try:
            self.video_processor.load_video()

            while True:
                frame = self.video_processor.decode_frame(skip_frames=2)
                if frame is None:
                    break

                frame = cv2.resize(frame, (frame_width, frame_height))

                control_frame = region_adjuster.draw_overlay(frame)  # Apply transparent red overlay
                region_adjuster.draw_boundary(control_frame)  # Draw the line and endpoints

                roi_masked = region_adjuster.apply_roi_mask(frame)

                motion_boxes = self.motion_detector.detect_motion(frame, roi_masked)
                detected_cars = self.car_detector.detect_moving_cars(frame, roi_masked, motion_boxes)
                self.license_plate_detector.detect_license_plates(frame, roi_masked, detected_cars)

                cv2.imshow("LPR-Control", control_frame)
                cv2.imshow("LPR-Main", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.video_processor.release_resources()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "recordings\\motion4.mp4"
    pipeline = LPRPipeline(video_path)
    pipeline.run()
