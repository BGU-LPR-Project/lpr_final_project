import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QListWidget, QHBoxLayout, QSizePolicy, QRadioButton, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont  
from PyQt5.QtCore import QTimer, Qt, QSize
from main import LPRPipeline
import os
# import qdarkstyle


from roi import RegionAdjuster
from cloud import CarDetector, LicensePlateDetector
from edge import MotionDetector
from collections import OrderedDict



ICON_DIR = os.path.join(os.path.dirname(__file__), "icons")


class LPRApp(QWidget):
    def __init__(self):
        super().__init__()

        self.video_path = None
        self.cap = None
        self.pipeline = None
        self.timer = QTimer(self)

        # Initialize detection and tracking
        self.car_detector = CarDetector("yolo11n.pt")
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "license_plate_detector.pt")
        self.license_plate_detector = LicensePlateDetector(MODEL_PATH)
        self.motion_detector = MotionDetector()

        self.region_adjuster = None  # Initialize later with video dimensions
        self.tracked_objects = OrderedDict()

        self.counts = {
            'entrance': 0,
            'exit': 0,
            'entrance_detected': 0,
            'exit_detected': 0,
            'total_counted': 0,
            'total_detected': 0
        }

        # Track IDs already counted
        self.counted_ids = set()  # cars counted once they appear
        self.plate_counted_ids = set()  # cars counted once their plate detected

        self.ui_mode = "REGULAR"

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("License Plate Recognition - LPR UI")
        self.setGeometry(100, 100, 1280, 720)
        self.setWindowIcon(QIcon(os.path.join(ICON_DIR, "lpr_icon.png")))

        # Main video display
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setScaledContents(True)
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self.video_mousePressEvent
        self.video_label.mouseMoveEvent = self.video_mouseMoveEvent
        self.video_label.mouseReleaseEvent = self.video_mouseReleaseEvent

        # Sidebar widgets
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setAlignment(Qt.AlignTop)
        sidebar_layout.setSpacing(10)
        sidebar_layout.setContentsMargins(10, 20, 10, 10)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setLayout(sidebar_layout)

        # Mode selection buttons
        self.region_btn = QPushButton("Region Selection")
        self.regular_btn = QPushButton("Regular Playback")
        self.detection_btn = QPushButton("Detection View")

        
        for btn in [self.region_btn, self.regular_btn, self.detection_btn]:
            sidebar_layout.addWidget(btn)

        # Statistics labels
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(QLabel("Counting Statistics:"))

        self.entrance_count_label = QLabel("Entrance Count: 0")
        self.entrance_detected_label = QLabel("Entrance Detected: 0")
        self.exit_count_label = QLabel("Exit Count: 0")
        self.exit_detected_label = QLabel("Exit Detected: 0")
        self.total_count_label = QLabel("Total Counted: 0")
        self.total_detected_label = QLabel("Total Detected: 0")

        stats_style = """
        background-color: #d2e5f5;
        padding: 4px;
        border-radius: 4px;
        color: #003B73;
        font-weight: bold;
        """

        for lbl in [self.entrance_count_label, self.entrance_detected_label,
                    self.exit_count_label, self.exit_detected_label,
                    self.total_count_label, self.total_detected_label]:
            lbl.setStyleSheet(stats_style)
            lbl.setContentsMargins(0, 2, 0, 2)
            # lbl.setStyleSheet("font-size: 14px; padding: 5px; color: white; font-weight: bold;")
            sidebar_layout.addWidget(lbl)

        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(QLabel("Currently Detected Plates:"))
        self.plates_list = QListWidget()
        self.plates_list.setStyleSheet("""
        QListWidget {
            background-color: #ffffff;
            border: 1px solid #c3d9ee;
            border-radius: 6px;
            color: #003B73;
            font-size: 10pt;
            padding: 6px;
        }
        """)
        sidebar_layout.addWidget(self.plates_list)

        # Horizontal layout (sidebar + video)
        main_layout = QHBoxLayout()
        main_layout.addWidget(sidebar, 2)  # Sidebar takes 20% width
        main_layout.addWidget(self.video_label, 8)  # Video takes 80% width

        # Bottom playback control buttons
        control_layout = QHBoxLayout()
        control_layout = QHBoxLayout()
        
        self.load_btn = QPushButton()
        self.play_btn = QPushButton()
        self.pause_btn = QPushButton()
        self.restart_btn = QPushButton()

        icon_size = QSize(48, 48)
        self.load_btn.setToolTip("Load video from your computer")
        self.load_btn.setIcon(QIcon(os.path.join(ICON_DIR, "load.png")))
        self.load_btn.setIconSize(icon_size)

        self.play_btn.setToolTip("Start video playback")
        self.play_btn.setIcon(QIcon(os.path.join(ICON_DIR, "play.png")))
        self.play_btn.setIconSize(icon_size)

        self.pause_btn.setToolTip("Pause video")
        self.pause_btn.setIcon(QIcon(os.path.join(ICON_DIR, "pause.png")))
        self.pause_btn.setIconSize(icon_size)

        self.restart_btn.setToolTip("Restart video from the beginning")
        self.restart_btn.setIcon(QIcon(os.path.join(ICON_DIR, "restart.png")))
        self.restart_btn.setIconSize(icon_size)


        for btn in [self.load_btn, self.play_btn, self.pause_btn, self.restart_btn]:
            control_layout.addWidget(btn)

        final_layout = QVBoxLayout()
        final_layout.addLayout(main_layout)
        final_layout.addLayout(control_layout)

        self.status_label = QLabel("Ready 🔍")
        self.status_label.setStyleSheet("font-size: 14px; color: #5E5E5E; padding: 8px; font-weight: bold;")
        final_layout.addWidget(self.status_label)
        self.setLayout(final_layout)

        # Connect button actions
        self.load_btn.clicked.connect(self.load_video)
        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)
        self.restart_btn.clicked.connect(self.restart_video)

        # Connect sidebar buttons
        self.region_btn.clicked.connect(lambda: self.set_ui_mode("REGION"))
        self.regular_btn.clicked.connect(lambda: self.set_ui_mode("REGULAR"))
        self.detection_btn.clicked.connect(lambda: self.set_ui_mode("DETECTION"))

        # Video timer
        self.timer.timeout.connect(self.next_frame)

        # Initialize mode
        self.ui_mode = "REGULAR"

    def update_plates_list(self):
        self.plates_list.clear()
        for obj in self.tracked_objects.values():
            plate_number = obj["plate_number"]
            if plate_number != "---":
                self.plates_list.addItem(f"{plate_number}")

    def restart_video(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.reset_count()

    def video_mousePressEvent(self, event):
        if self.ui_mode == "REGION":
            x = int(event.pos().x() * self.region_adjuster.frame_width / self.video_label.width())
            y = int(event.pos().y() * self.region_adjuster.frame_height / self.video_label.height())
            self.region_adjuster.select_boundary(cv2.EVENT_LBUTTONDOWN, x, y, None, None)
            self.refresh_display_frame()

    def video_mouseMoveEvent(self, event):
        if self.ui_mode == "REGION":
            x = int(event.pos().x() * self.region_adjuster.frame_width / self.video_label.width())
            y = int(event.pos().y() * self.region_adjuster.frame_height / self.video_label.height())
            self.region_adjuster.select_boundary(cv2.EVENT_MOUSEMOVE, x, y, None, None)
            self.refresh_display_frame()

    def video_mouseReleaseEvent(self, event):
        if self.ui_mode == "REGION":
            x = int(event.pos().x() * self.region_adjuster.frame_width / self.video_label.width())
            y = int(event.pos().y() * self.region_adjuster.frame_height / self.video_label.height())
            self.region_adjuster.select_boundary(cv2.EVENT_LBUTTONUP, x, y, None, None)
            self.refresh_display_frame()

    def refresh_display_frame(self):
        if self.cap and self.cap.isOpened():
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                display_frame = frame.copy()

                if self.ui_mode == "REGION":
                    display_frame = self.region_adjuster.draw_overlay(display_frame)
                    self.region_adjuster.draw_boundary(display_frame)
                    self.region_adjuster.draw_labels(display_frame)

                # Convert and display immediately
                display_frame = cv2.resize(display_frame, (1024, 576), interpolation=cv2.INTER_AREA)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = display_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def set_ui_mode(self, mode):
        self.ui_mode = mode
        print(f"UI mode set to: {mode}")

        # Immediate visual refresh if paused
        if self.cap and self.cap.isOpened() and not self.timer.isActive():
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)  # reset position immediately
                display_frame = frame.copy()

                # Draw according to the new mode immediately
                if mode == "REGION":
                    display_frame = self.region_adjuster.draw_overlay(display_frame)
                    self.region_adjuster.draw_boundary(display_frame)
                    self.region_adjuster.draw_labels(display_frame)

                elif mode == "DETECTION":
                    # Use last detection data (no need to run detection again)
                    for car_id, car_data in self.detected_cars.items():
                        bbox = car_data["bbox"]
                        x1, y1, x2, y2 = bbox
                        centroid = car_data["centroid"]

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                        plate_number = "---"
                        confidence = 0.0
                        if car_id in self.tracked_objects:
                            plate_number = self.tracked_objects[car_id]["plate_number"]
                            confidence = self.tracked_objects[car_id]["confidence"]

                        cv2.putText(display_frame, f"ID: {car_id}", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                        if plate_number != "---":
                            cv2.putText(display_frame, f"{plate_number} ({confidence:.2f})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                # Convert and display immediately
                display_frame = cv2.resize(display_frame, (1024, 576), interpolation=cv2.INTER_AREA)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = display_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))

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

            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                self.region_adjuster = RegionAdjuster(w, h)
                self.reset_count()

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.status_label.setText("Video loaded successfully ✔️")
        else:
            self.status_label.setText("No video file selected ❌")

    def play_video(self):
        if self.cap and self.cap.isOpened():
            self.timer.start(30)
            self.status_label.setText("Playing video ▶️")

    def pause_video(self):
        if self.pipeline:
            self.pipeline.pause()
        self.timer.stop()
        self.status_label.setText("Video paused ⏸️")

    def reset_count(self):
        self.counted_ids.clear()
        self.plate_counted_ids.clear()
        self.counts = {
            'entrance': 0,
            'exit': 0,
            'entrance_detected': 0,
            'exit_detected': 0,
            'total_counted': 0,
            'total_detected': 0
        }
        self.update_counts_display()
        self.update_plates_list()
        self.tracked_objects.clear()
        self.status_label.setText("Video restarted 🔄")

    def next_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                display_frame = frame.copy()

                # Always perform detection logic (in all modes!)
                roi_masked = self.region_adjuster.apply_roi_mask(frame)
                motion_boxes = self.motion_detector.detect_motion(display_frame, roi_masked)
                detected_cars = self.car_detector.detect_moving_cars(
                    display_frame, roi_masked, motion_boxes, self.region_adjuster.is_in_entrance_or_exit
                )

                # Store all detected cars
                self.detected_cars = detected_cars

                # Detect plates on these cars
                self.tracked_objects = self.license_plate_detector.detect_license_plates(
                    display_frame, roi_masked, detected_cars, self.region_adjuster.is_in_entrance_or_exit
                )

                # Always update counting (background counting)
                self.update_counts()

                # Conditional Visualization (only in DETECTION mode)
                if self.ui_mode == "DETECTION":
                    for car_id, car_data in detected_cars.items():
                        bbox = car_data["bbox"]
                        x1, y1, x2, y2 = bbox
                        centroid = car_data["centroid"]

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                        plate_number = "---"
                        confidence = 0.0
                        if car_id in self.tracked_objects:
                            plate_number = self.tracked_objects[car_id]["plate_number"]
                            confidence = self.tracked_objects[car_id]["confidence"]

                        cv2.putText(display_frame, f"ID: {car_id}", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                        if plate_number != "---":
                            cv2.putText(display_frame, f"{plate_number} ({confidence:.2f})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                # Visualization for REGION mode
                if self.ui_mode == "REGION":
                    display_frame = self.region_adjuster.draw_overlay(display_frame)
                    self.region_adjuster.draw_boundary(display_frame)
                    self.region_adjuster.draw_labels(display_frame)

                # For REGULAR mode, no additional drawing needed (clean frame)

                # Convert and display the frame
                display_frame = cv2.resize(display_frame, (1024, 576), interpolation=cv2.INTER_AREA)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = display_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_counts(self):
        for car_id, car_data in self.detected_cars.items():
            direction = car_data["direction"]

            # Count cars only once when first detected
            if car_id not in self.counted_ids:
                self.counts['total_counted'] += 1
                if direction == "Entrance":
                    self.counts['entrance'] += 1
                elif direction == "Exit":
                    self.counts['exit'] += 1
                self.counted_ids.add(car_id)

            # Separately count detected plates if plate is now detected and wasn't counted before
            tracked_obj = self.tracked_objects.get(car_id)
            if (tracked_obj and
                    tracked_obj.get("plate_number") not in [None, '---', 'UNKNOWN'] and
                    car_id not in self.plate_counted_ids):

                self.counts['total_detected'] += 1
                if direction == "Entrance":
                    self.counts['entrance_detected'] += 1
                elif direction == "Exit":
                    self.counts['exit_detected'] += 1
                self.plate_counted_ids.add(car_id)

        # Update labels and plate list
        self.update_counts_display()
        self.update_plates_list()

    def update_counts_display(self):
        self.entrance_count_label.setText(f"Entrance Count: {self.counts['entrance']}")
        self.entrance_detected_label.setText(f"Entrance Detected: {self.counts['entrance_detected']}")
        self.exit_count_label.setText(f"Exit Count: {self.counts['exit']}")
        self.exit_detected_label.setText(f"Exit Detected: {self.counts['exit_detected']}")
        self.total_count_label.setText(f"Total Counted: {self.counts['total_counted']}")
        self.total_detected_label.setText(f"Total Detected: {self.counts['total_detected']}")

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
    app.setFont(QFont("Segoe UI", 10))
    
    light_style = """
    QWidget {
        background-color: #eaf4fb;
        color: #003B73;
        font-family: 'Segoe UI';
        font-size: 10pt;
    }

    QPushButton {
        background-color: #d4eafd;
        color: #003B73;
        font-weight: 600;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #a8cdee;
    }

    QPushButton:hover {
        background-color: #e4f4ff;
    }

    QPushButton:pressed {
        background-color: #bdddf5;
    }

    QLabel {
        font-size: 12pt;
        font-weight: 600;
        color: #003B73;
    }

    QListWidget {
        background-color: #ffffff;
        border: 1px solid #c3d9ee;
        border-radius: 6px;
        color: #003B73;
        font-size: 10pt;
        padding: 6px;
    }

    QWidget#sidebar QLabel {
    font-size: 11pt;
    font-weight: 500;
    color: #004a88;
    }

    QWidget#sidebar QLabel:first-child {
        font-size: 13pt;
        font-weight: bold;
        margin-bottom: 10px;
    }

    QToolTip {
        background-color: #f2fbff;
        color: #003B73;
        border: 1px solid #a0cbe8;
        padding: 5px;
        border-radius: 6px;
        font-size: 9pt;
    }

    QWidget#sidebar {
        background-color: #dff0fa;
        border-right: 2px solid #bddff2;
        padding: 12px;
    }
    """
    app.setStyleSheet(light_style)

    window = LPRApp()
    window.show()
    sys.exit(app.exec_())
