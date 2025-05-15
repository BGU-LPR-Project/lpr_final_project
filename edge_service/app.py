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
from queue import Queue

VISUAL_FRAME_QUEUE = "visual_frame_queue"

COOLDOWN = threading.Event()

# Constants
FRAME_QUEUE = Queue(maxsize=30)  # Limit to control memory usage

# FastAPI app initialization
app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def trigger_cooldown(edge_service):
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
    """Connect to the Redis server and wait for it to be ready."""
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
    """Process each frame using edge service and send results to cloud."""
    print("Processing frame in worker thread.")
    result = {}

    def callback(prediction):
        nonlocal result
        result = prediction

    edge_service.predict(frame, callback)

    for object_id, box in result.items():
        try:
            cropped_plate = frame[box[1]:box[3], box[0]:box[2]]
            _, buffer = cv2.imencode('.jpg', cropped_plate)
            encoded_plate = base64.b64encode(buffer).decode('utf-8')

            cloud_response = requests.post(
                "http://cloud_service:8000/predict",
                json={"plate_img": encoded_plate},
                verify=True
            ).json()

            ocr_text, ocr_conf = cloud_response.get("ocr_result", (str(), 0.0))
            edge_service.update_tracked_vehicle(object_id, ocr_text, ocr_conf)

            # Trigger cooldown after successful recognition
            # trigger_cooldown(edge_service)
        except Exception as e:
            print(f"cloud predict api failed: {e}")
    annotated_frame = edge_service.visualize(frame, True)
    redis_client.rpush(VISUAL_FRAME_QUEUE, pickle.dumps(annotated_frame))


def poll_queue(redis_client, edge_service):
    """Poll the Redis queue and enqueue frames for processing."""
    while True:
        frame_data = redis_client.lpop("frame_queue")
        if frame_data:
            frame = pickle.loads(frame_data)
            try:
                FRAME_QUEUE.put(frame, timeout=0.1)
            except:
                print("Frame queue is full. Dropping frame.")
        # edge_service.log_results()
        # time.sleep(0.1)


def frame_worker(edge_service):
    """Worker thread to consume frames and process them."""
    while True:
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

@app.get("/healthcheck")
async def healthcheck():
    return "Edge service running!", 200

def main():
    global redis_client
    redis_client = connect_to_redis()
    edge_service = EdgeService("/app/models/yolo11n.pt", "/app/models/license_plate_detector.pt")
    edge_service.on()

    # Start the worker thread and polling thread
    NUM_WORKERS = 4  # Adjust based on your CPU and memory capacity

    # Start multiple frame processing workers
    for _ in range(NUM_WORKERS):
        threading.Thread(target=frame_worker, args=(edge_service,), daemon=True).start()
    
    threading.Thread(target=poll_queue, args=(redis_client, edge_service), daemon=True).start()

    # Run FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
