import cv2

class VideoHandler:
    def __init__(self, video_path: str):
        """
        Initialize the VideoProcessor with the path to the video file.
        """
        self.video_path: str = video_path
        self.capture = None

    def load_video(self) -> None:
        """
        Load the video file and check if it is accessible.
        """
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {self.video_path}")

    def decode_frame(self):
        """
        Decode a single frame from the video. Returns None if no frame is available.
        """
        if self.capture is None:
            raise ValueError("Video capture not initialized. Call load_video() first.")
        
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame

    def release_resources(self) -> None:
        """
        Release the video file resources properly.
        """
        if self.capture:
            self.capture.release()
        print("Video processing stopped. Resources released.")
