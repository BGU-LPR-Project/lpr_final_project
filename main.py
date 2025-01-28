from typing import Optional, Dict
import cv2
from datetime import datetime
from video_handler import VideoHandler
from edge import *
from cloud import *
from roi import RegionAdjuster

class LicensePlate:
    """
    Represents a license plate instance.
    """
    def __init__(self, plate_number: str):
        self.plate_number = plate_number
        self.first_time_seen = datetime.now()
        self.last_time_seen = datetime.now()
        # self.perimeter = perimeter

    def update_last_seen(self):
        """
        Updates the last time the license plate was seen.
        """
        self.last_time_seen = datetime.now()

class LPRPipeline:
    def __init__(self, video_path: str):
        self.video_processor = VideoHandler(video_path)
        self.motion_detector = MotionDetector()  # Make sure it uses the updated MotionDetector
        self.car_detector = CarDetector("yolo11n.pt")
        self.license_plate_detector = LicensePlateDetector("license_plate_detector.pt", [
            r'[A-Z]{2}[0-9]{2}[A-Z]{3}',
            r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}',
            r'[А-Я]{1,2}[0-9]{3}[А-Я]{1,2}[0-9]{2,3}',
            r'[1-9A-Z]{2}[A-Z0-9]*[-]{1}[0-9]{4}',
            r'[A-Z]{1,2}[0-9]{3}[A-Z]{1,2}[0-9]{2,3}', 
            r'[А-Я]{1}[0-9]{3}[А-Я]{2}[0-9]{2}',
            r'[1-9A-Z]{1}[A-Z]{3}[0-9]{3}',
            r'[1-9A-Z]{1}[A-Z]{5}',
            r'^[1-9][A-Z]{3}[0-9]{3}$', #California License Plate
            r'[A-Z]{1}[0-9]{3}[A-Z]{2}[0-9]{3}'
        ])
        self.license_plate_table: Dict[str, LicensePlate] = {}  # Hash table for license plates

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

                frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

                control_frame = region_adjuster.draw_overlay(frame)  # Apply transparent red overlay
                region_adjuster.draw_boundary(control_frame)  # Draw the line and endpoints
                region_adjuster.draw_labels(control_frame)
                
                roi_masked = region_adjuster.apply_roi_mask(frame)

                motion_boxes = self.motion_detector.detect_motion(frame, roi_masked)
                detected_cars = self.car_detector.detect_moving_cars(frame, roi_masked, motion_boxes)
                detections = self.license_plate_detector.detect_license_plates(frame, roi_masked, detected_cars)

                # Go over the detections
                for object_id, data in detections.items():
                    plate_number = data["plate_number"]
                    if plate_number in self.license_plate_table:
                            # Update existing license plate instance
                            self.license_plate_table[plate_number].update_last_seen()
                            print(f"Updated: {plate_number} | Last seen: {self.license_plate_table[plate_number].last_time_seen}")
                    else:
                        # Create a new license plate instance
                        self.license_plate_table[plate_number] = LicensePlate(plate_number)
                        print(f"New plate: {plate_number} | First seen: {self.license_plate_table[plate_number].first_time_seen}")
                    authorized = False #TODO: Call Whitelist component to check plate_number
                    self.visualize(frame, object_id, data, authorized)

                # Display the processed frame
                cv2.imshow("LPR-Control", control_frame)
                cv2.imshow("LPR-Main", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.video_processor.release_resources()
            cv2.destroyAllWindows()

    def visualize(self, frame: np.ndarray, object_id: int, data, authorized: bool):
        centroid = data["centroid"]
        bbox = data["bbox"]
        plate_number = data["plate_number"]

        box_color = (0, 255, 0) if authorized else (0, 0, 255)

        # Draw the bounding box and centroid
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, tuple(centroid), 5, box_color, -1)
        text = f"ID {object_id}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Optionally display the plate number if available
        if plate_number:
            cv2.putText(frame, f"Plate: {plate_number}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

if __name__ == "__main__":
    video_path = "recordings\\motion4.mp4"
    pipeline = LPRPipeline(video_path)
    pipeline.run()

    print("\nSummary of all detected license plates:")
    for plate_number, plate_data in pipeline.license_plate_table.items():
        print(f"Plate: {plate_number} | First seen: {plate_data.first_time_seen} | Last seen: {plate_data.last_time_seen}")
