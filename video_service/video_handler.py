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

    def decode_frame(self, skip_frames: int = 0):
        """
        Decode a single frame from the video, skipping a given number of frames.
        """
        if self.capture is None:
            raise ValueError("Video capture not initialized. Call load_video() first.")

        for _ in range(skip_frames):
            self.capture.grab()  # Skip frames without decoding them

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
