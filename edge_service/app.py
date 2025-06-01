import base64
import cv2
import redis
import threading
import time
import requests
import pickle
import uvicorn
import logging
from fastapi import FastAPI
from edge import EdgeService
from queue import Queue, Empty

VISUAL_FRAME_QUEUE = "visual_frame_queue"

COOLDOWN = threading.Event()  # Prevents repeated edge triggers
PAUSED = threading.Event()    # Used to pause processing

# Constants
FRAME_QUEUE = Queue(maxsize=30)  # Bounded queue to control memory usage

# FastAPI app initialization
app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def trigger_cooldown(edge_service):
    """Triggers a cooldown period where the edge service is briefly reset."""
    if not COOLDOWN.is_set():
        def cooldown_logic():
            print("Cooldown started.")
            COOLDOWN.set()
            edge_service.off()
            time.sleep(1)
            edge_service.on()
            COOLDOWN.clear()
            print("Cooldown ended.")

        threading.Thread(target=cooldown_logic, daemon=True).start()

def connect_to_redis():
    """Connects to Redis and waits until it's ready to accept commands."""
    client = redis.StrictRedis(host='redis', port=6379, db=0)
    while True:
        try:
            client.ping()
            break
        except redis.exceptions.BusyLoadingError:
            print("Waiting for Redis to load data into memory...")
            time.sleep(1)
    return client

def process_frame(frame, edge_service):
    """Runs prediction on a frame, sends to cloud for OCR, and pushes visual output."""
    print("Processing frame in worker thread.")
    result = {}

    def callback(prediction):
        nonlocal result
        result = prediction

    edge_service.predict(frame, callback)

    for object_id, box in result.items():
        try:
            # Extract plate region
            cropped_plate = frame[box[1]:box[3], box[0]:box[2]]
            _, buffer = cv2.imencode('.jpg', cropped_plate)
            encoded_plate = base64.b64encode(buffer).decode('utf-8')

            # Send to cloud OCR service
            cloud_response = requests.post(
                "http://cloud_service:8000/predict",
                json={"plate_img": encoded_plate},
                verify=True
            ).json()

            ocr_text, ocr_conf = cloud_response.get("ocr_result", (str(), 0.0))
            edge_service.update_tracked_vehicle(object_id, ocr_text, ocr_conf)

            # Optional: cooldown after successful recognition
            # trigger_cooldown(edge_service)
        except Exception as e:
            print(f"cloud predict api failed: {e}")

    # Push visualized output to Redis for frontend use
    annotated_frame = edge_service.visualize(frame, True)
    redis_client.rpush(VISUAL_FRAME_QUEUE, pickle.dumps(annotated_frame))

def poll_queue(redis_client, edge_service):
    """Continuously polls Redis for new frames and enqueues them for processing."""
    while True:
        frame_data = redis_client.lpop("frame_queue")
        if frame_data:
            frame = pickle.loads(frame_data)
            try:
                FRAME_QUEUE.put(frame, timeout=0.1)
            except:
                print("Frame queue is full. Dropping frame.")

def frame_worker(edge_service):
    """Consumes frames from queue and processes them using the edge service."""
    while True:
        if not PAUSED.is_set():
            frame = FRAME_QUEUE.get()
            if COOLDOWN.is_set():
                print("Cooldown active. Dropping frame.")
                FRAME_QUEUE.task_done()
                continue
            try:
                process_frame(frame, edge_service)
            except Exception as e:
                print(f"Error processing frame: {e}")
            finally:
                FRAME_QUEUE.task_done()
        else:
            time.sleep(1)

def clear_thread_queue():
    """Clears all frames from the thread-safe processing queue."""
    while not FRAME_QUEUE.empty():
        try:
            FRAME_QUEUE.get_nowait()
        except Empty:
            break

@app.get("/edge-pause")
async def edge_pause():
    """Pauses edge frame processing."""
    PAUSED.set()
    return "Edge paused!", 200

@app.get("/edge-resume")
async def edge_resume():
    """Resumes edge frame processing."""
    PAUSED.clear()
    return "Edge resumed!", 200

@app.get("/clear-queue")
async def clear_queues():
    """Clears both Redis queues and the in-memory frame queue."""
    redis_client.ltrim("frame_queue", 1, 0)
    redis_client.ltrim("visual_frame_queue", 1, 0)
    clear_thread_queue()

@app.get("/skip-at-most-ten")
async def skip():
    """Skips up to 10 seconds of processing by clearing 40 frames."""
    max_frames_to_skip = 40
    frames_skipped = 0

    while frames_skipped < max_frames_to_skip and not FRAME_QUEUE.empty():
        try:
            FRAME_QUEUE.get_nowait()
            FRAME_QUEUE.task_done()
            frames_skipped += 1
        except Empty:
            break

    seconds_skipped = frames_skipped // 4  # Assumes 4 FPS
    return {"seconds_skipped": seconds_skipped}

@app.get("/healthcheck")
async def healthcheck():
    """Checks if the edge service is running."""
    return "Edge service running!", 200

def main():
    """Initializes services, starts worker threads, and runs the FastAPI app."""
    global redis_client
    redis_client = connect_to_redis()
    edge_service = EdgeService("/app/models/yolo11n.pt", "/app/models/license_plate_detector.pt")
    edge_service.on()

    # Start the worker thread and polling thread
    NUM_WORKERS = 4  # Adjust based on your CPU and memory capacity

    for _ in range(NUM_WORKERS):
        threading.Thread(target=frame_worker, args=(edge_service,), daemon=True).start()
    
    threading.Thread(target=poll_queue, args=(redis_client, edge_service), daemon=True).start()

    # Run FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
