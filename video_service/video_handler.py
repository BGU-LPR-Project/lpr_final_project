import cv2

class VideoHandler:
    def __init__(self, video_path_or_stream, target_fps=4):
        self.video_path_or_stream = video_path_or_stream
        self.cap = None
        self.target_fps = target_fps
        self.frame_skip_interval = 1
        self.frame_count = 0

    def load_video(self):
        self.cap = cv2.VideoCapture(self.video_path_or_stream)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video or stream: {self.video_path_or_stream}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 25  # fallback default

        self.frame_skip_interval = int(fps / self.target_fps)
        if self.frame_skip_interval < 1:
            self.frame_skip_interval = 1

        print(f"[VideoHandler] FPS: {fps:.2f}, skipping every {self.frame_skip_interval - 1} frames")

    def decode_frame(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return None

            self.frame_count += 1
            if self.frame_count % self.frame_skip_interval == 0:
                return frame  # only process this frame

    def seek(self, seconds: float):
        if self.cap and not self.video_path_or_stream.startswith("rtsp://"):
            current_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.cap.set(cv2.CAP_PROP_POS_MSEC, current_msec + seconds * 1000)

    def release_resources(self):
        if self.cap:
            self.cap.release()
