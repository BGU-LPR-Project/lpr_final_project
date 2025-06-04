# 🚗 LPR - License Plate Recognition System

A smart parking enforcement solution using AI-powered license plate recognition (LPR).  
This project enables real-time detection of unauthorized vehicles using existing security cameras. It includes motion detection, vehicle tracking, license plate recognition, and real-time alerts.

---

## 🔄 Flow Explanation

1. **VideoService** reads a local `.mp4` video and streams frames into Redis.
2. **EdgeService** receives frames, detects motion and vehicles using YOLOv8, and matches bounding boxes to potential license plate regions.
3. **CloudService** optionally performs OCR on plates and matches them against an authorized list.
4. **UI** displays the processed video and alerts when an unauthorized vehicle is detected.

---

## 📦 Project Structure
lpr_final_project/
├── cloud_service/ # Cloud OCR and list matching
├── edge_service/ # Motion & vehicle detection
├── video_service/ # Reads and streams video frames
├── models/ # YOLO pretrained models
├── recordings/ # Local test videos (e.g., car.mp4)
├── tests/ # Testing
├── ui.py # PyQt5 UI for monitoring
├── docker-compose.yml # Multi-container setup
├── main.py
└── README.md

## ⚙️ Tech Stack

- **YOLOv11 (Ultralytics)** – Object detection for cars and plates
- **PaddleOCR** – License plate text recognition
- **Redis** – Frame queueing and communication
- **FastAPI** – Lightweight backend API
- **PyQt5** – Desktop UI for visualizing results and debugging
- **Docker + Docker Compose** – For container orchestration

## 🚀 Setup Instructions

### Prerequisites
- Docker & Docker Compose installed
- Python 3.9+ (for running `ui.py` locally)
- Place a test video (e.g., `car.mp4`) inside:  
  `lpr_final_project/recordings/`

### 1. Clone the Repo
```bash command: 
git clone https://github.com/BGU-LPR-Project/lpr_final_project.git
cd lpr_final_project
```
### 2. Build and Start Docker Services
```bash
docker-compose build
docker-compose up
```
expected:
✔ cloud_service  Built
                                                                
✔ edge_service   Built
                                                                
✔ video_service  Built
                       

# 3. Run the UI (in a new terminal)
```bash
python ui.py
```