import cv2
import time
import threading

class VideoHandler:
    def __init__(self, video_path_or_stream, target_fps=4):
        """
        Initializes the video handler for either a video file or an RTSP stream.
        :param video_path_or_stream: Path to video file or RTSP stream URL
        :param target_fps: Target FPS for processing (default 4)
        """
        self.video_path_or_stream = video_path_or_stream
        self.cap = None
        self.target_fps = target_fps
        self.last_frame_time = time.time()

        # Mutex lock for thread safety
        self.lock = threading.Lock()

    def load_video(self):
        """Loads the video or RTSP stream depending on the provided path."""
        if self.video_path_or_stream.startswith("rtsp://"):
            self.cap = cv2.VideoCapture(self.video_path_or_stream)
        else:
            self.cap = cv2.VideoCapture(self.video_path_or_stream)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video or stream: {self.video_path_or_stream}")

    def decode_frame(self):
        """
        Decodes a single frame from the video or stream.
        Returns None if the video or stream ends.
        """
        with self.lock:  # Ensure only one thread interacts with the video capture
            ret, frame = self.cap.read()
            if not ret:
                return None  # End of video or stream

            # Throttle the frame rate to match the target FPS
            current_time = time.time()
            time_since_last_frame = current_time - self.last_frame_time

            # If enough time has passed since the last frame, return the current frame
            if time_since_last_frame >= (1 / self.target_fps):
                self.last_frame_time = current_time
                return frame
            else:
                # If not enough time has passed, skip this frame and return None
                return None
        
    def seek(self, seconds: float):
        """Seeks forward by a number of seconds (only for file, not RTSP)."""
        if self.cap is None or self.video_path_or_stream.startswith("rtsp://"):
            return  # Skip seeking for live streams

        with self.lock:  # Ensure only one thread interacts with the video capture during seeking
            # Get the current time in milliseconds
            current_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            
            if current_msec == 0:
                print("Error: Unable to get current position.")
                return
            
            new_msec = current_msec + (seconds * 1000)  # Seek in milliseconds
            
            # Ensure seeking within bounds
            self.cap.set(cv2.CAP_PROP_POS_MSEC, new_msec)
            
            # Add a small delay to give FFmpeg time to buffer and handle seeking properly
            time.sleep(1)  # Sleep for 1 second to allow stream to stabilize
            
            # Check if the video is still open and if the position was set correctly
            if not self.cap.isOpened():
                print("Error: Video stream closed after seek.")
                return
            
            print(f"Successfully skipped {seconds} seconds.")
        
    def release_resources(self):
        """Releases the video capture object."""
        if self.cap is not None:
            self.cap.release()
