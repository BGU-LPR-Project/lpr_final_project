import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QListWidget, QHBoxLayout, QSizePolicy, QRadioButton, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import QTimer, Qt
from main import LPRPipeline
import os

ICON_DIR = os.path.abspath("icons")


class LPRApp(QWidget):
    def __init__(self):
        super().__init__()

        self.video_path = None
        self.cap = None
        self.pipeline = None
        self.timer = QTimer(self)
        self.processing_mode = "Real-Time"

        # UI Layout
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("License Plate Recognition - LPR UI")
        self.setGeometry(100, 100, 1024, 768)

        self.setWindowIcon(QIcon(os.path.join(ICON_DIR, "lpr_icon.png")))

        # Video Display
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setScaledContents(True)

        # Mode Selection
        self.mode_label = QLabel("Processing Mode:")
        self.real_time_btn = QRadioButton("Real-Time Processing")
        self.batch_btn = QRadioButton("Batch Processing")
        self.real_time_btn.setChecked(True)

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.real_time_btn)
        self.mode_group.addButton(self.batch_btn)

        self.real_time_btn.toggled.connect(lambda: self.set_processing_mode("Real-Time"))
        self.batch_btn.toggled.connect(lambda: self.set_processing_mode("Batch"))

        # Buttons
        self.load_btn = QPushButton(" Load Video")
        self.load_btn.setIcon(QIcon(os.path.join(ICON_DIR, "load.png")))

        self.play_btn = QPushButton("Play")
        self.play_btn.setIcon(QIcon(os.path.join(ICON_DIR, "play.png")))
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setIcon(QIcon(os.path.join(ICON_DIR, "pause.png")))

        # Button Styling
        button_style = """
            QPushButton {
                background-color: #2E3B4E;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3F4D60;
            }
            QPushButton:pressed {
                background-color: #1F2A3A;
            }
        """
        self.load_btn.setStyleSheet(button_style)
        self.play_btn.setStyleSheet(button_style)
        self.pause_btn.setStyleSheet(button_style)

        self.load_btn.clicked.connect(self.load_video)
        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)

        # Detected License Plates List
        self.plate_list = QListWidget()

        # Layouts
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(self.pause_btn)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.real_time_btn)
        mode_layout.addWidget(self.batch_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(mode_layout)
        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("Detected License Plates:"))
        layout.addWidget(self.plate_list)

        self.setLayout(layout)

        # Timer for playing video
        self.timer.timeout.connect(self.next_frame)

    def set_processing_mode(self, mode):
        """Updates the processing mode based on user selection."""
        self.processing_mode = mode
        print(f"Processing Mode Set To: {mode}")

    def load_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            self.pipeline = LPRPipeline(file_path)  # Initialize LPR processing

            if self.processing_mode == "Batch":
                self.process_full_video()

    def play_video(self):
        if self.cap and self.cap.isOpened():
            self.timer.start(30)

    def pause_video(self):
        if self.pipeline:
            self.pipeline.pause()
        self.timer.stop()

    def next_frame(self):
        if self.pipeline and self.processing_mode == "Real-Time":
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Resize and convert for UI display
                    frame = cv2.resize(frame, (1024, 576), interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(q_img))

                    # Run LPR detection
                    detected_plates = self.pipeline.run_ui_mode(frame)
                    
                    # Update UI with detected plates
                    self.plate_list.clear()
                    for plate in detected_plates:
                        self.plate_list.addItem(f"Plate: {plate}")

                else:
                    self.pipeline.log_detection_results()
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def process_full_video(self):
        """Processes entire video in batch mode."""
        print("Processing Full Video...")
        plates_detected = self.pipeline.run()
        
        self.plate_list.clear()
        for plate in plates_detected:
            self.plate_list.addItem(f"Plate: {plate}")

        self.pipeline.log_detection_results()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LPRApp()
    window.show()
    sys.exit(app.exec_())
