from typing import Dict
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
    def __init__(self, plate_number: str, direction: str):
        self.plate_number = plate_number
        self.first_time_seen = datetime.now()
        self.last_time_seen = datetime.now()
        self.direction = direction

    def update(self, plate_number: str, direction: str):
        self.last_time_seen = datetime.now()
        self.plate_number = plate_number
        self.direction = direction

class LPRPipeline:
    def __init__(self, video_path: str):
        self.video_processor = VideoHandler(video_path)
        self.motion_detector = MotionDetector()  # Make sure it uses the updated MotionDetector
        self.car_detector = CarDetector("yolo11n.pt")
        self.license_plate_detector = LicensePlateDetector("license_plate_detector.pt")
        self.license_plate_table: Dict[int, LicensePlate] = {}  # Hash table for license plates
        self.paused = False  # Flag to track pause state

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
                if not self.paused:
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
                    detections = self.license_plate_detector.detect_license_plates(frame, roi_masked, detected_cars, region_adjuster.is_in_entrance_or_exit)

                    # Go over the detections
                    for object_id, data in detections.items():
                        plate_number = data["plate_number"]
                        direction = data["direction"]

                        if object_id in self.license_plate_table:
                            # Update existing license plate instance
                            self.license_plate_table[object_id].update(plate_number, direction)
                        else:
                            # Create a new license plate instance
                            self.license_plate_table[object_id] = LicensePlate(plate_number, direction)
                            
                        authorized = plate_number != "---" #TODO: Call Whitelist component to check plate_number
                        self.visualize(frame, object_id, data, authorized)

                    # Display the processed frame
                    cv2.imshow("LPR-Control", control_frame)
                    cv2.imshow("LPR-Main", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord(' '):  # Pause/Resume with Spacebar
                    self.paused = not self.paused

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.video_processor.release_resources()
            cv2.destroyAllWindows()


    def visualize(self, frame: np.ndarray, object_id: int, data, authorized: bool):
        centroid = data["centroid"]
        bbox = data["bbox"]
        plate_number = data["plate_number"]
        plate_confidence = data["confidence"]
        direction = data["direction"]

        box_color = (0, 255, 0) if authorized else (0, 0, 255)

        # Draw the bounding box and centroid
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, tuple(centroid), 5, box_color, -1)
        text = f"ID {object_id} - {direction}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Optionally display the plate number if available
        if plate_number:
            cv2.putText(frame, f"Plate: {plate_number} - {plate_confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
    def logDetectionResults(self):
        print("\nSummary of all detected vehicles:")
        for vehicle_id, data in pipeline.license_plate_table.items():
            print(f"Vehicle-ID: {vehicle_id} | Plate: {data.plate_number} | First seen: {data.first_time_seen} | Last seen: {data.last_time_seen} | Direction: {data.direction}")


if __name__ == "__main__":
    video_path = "recordings\\motion4.mp4"
    pipeline = LPRPipeline(video_path)
    pipeline.run()
    pipeline.logDetectionResults()