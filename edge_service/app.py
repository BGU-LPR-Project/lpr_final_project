import base64
import cv2
import redis
import threading
import time
import requests
import pickle
import uvicorn
import logging
from fastapi import FastAPI, Request, HTTPException
from edge import EdgeService
from queue import Queue, Empty
from itertools import count
import concurrent.futures
import json

VISUAL_FRAME_QUEUE = "visual_frame_queue"

COOLDOWN = threading.Event()  # Prevents repeated edge triggers
PAUSED = threading.Event()    # Used to pause processing

FRAME_QUEUE = Queue(maxsize=30)  # Bounded queue to control memory usage
frame_counter = count()

# FastAPI app initialization
app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Thread pool executor for offloading OCR calls
OCR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def trigger_cooldown(edge_service):
    """Triggers a cooldown period where the edge service is briefly reset."""
    if not COOLDOWN.is_set():
        def cooldown_logic():
            logging.info("Cooldown started.")
            COOLDOWN.set()
            edge_service.off()
            time.sleep(1)
            edge_service.on()
            COOLDOWN.clear()
            logging.info("Cooldown ended.")

        threading.Thread(target=cooldown_logic, daemon=True).start()

def connect_to_redis():
    """Connects to Redis and waits until it's ready to accept commands."""
    client = redis.StrictRedis(host='redis', port=6379, db=0)
    while True:
        try:
            client.ping()
            break
        except redis.exceptions.BusyLoadingError:
            logging.info("Waiting for Redis to load data into memory...")
            time.sleep(1)
    return client

def call_ocr(encoded_plate):
    try:
        response = requests.post(
            "http://cloud_service:8000/predict",
            json={"plate_img": encoded_plate},
            verify=True,
            timeout=2
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.warning(f"OCR request failed: {e}")
        return None

def stop_video_if_visual_queue_stuck():
    MAX_QUEUE_SIZE = 15
    VIDEO_SERVICE_STOP_URL = "http://video_service:8000/stop-video"

    queue_size = redis_client.llen(VISUAL_FRAME_QUEUE)
    if queue_size >= MAX_QUEUE_SIZE:
        print(f"[WARN] Queue size reached {queue_size}, triggering stop-video.")
        try:
            response = requests.post(VIDEO_SERVICE_STOP_URL)
            print(f"[INFO] stop-video response: {response.status_code} - {response.json()}")
        except requests.RequestException as e:
            print(f"[ERROR] Could not reach video_service: {e}")
        return True
    return False

def push_detection_to_redis(detection):
    redis_client.rpush('DETECTIONS', json.dumps(detection))

def process_frame(seq, frame, edge_service):
    logging.info(f"Processing frame {seq} in worker thread.")
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

            # Synchronous OCR call
            cloud_response = call_ocr(encoded_plate)

            if cloud_response:
                ocr_text, ocr_conf = cloud_response.get("ocr_result", ("", 0.0))
                edge_service.update_tracked_vehicle(object_id, ocr_text, ocr_conf, frame, push_detection_to_redis)
            else:
                logging.warning(f"No OCR response for frame {seq}, object {object_id}")

            # Optionally trigger cooldown after recognition if needed
            # trigger_cooldown(edge_service)

        except Exception as e:
            logging.error(f"Cloud predict API failed: {e}")

    if not stop_video_if_visual_queue_stuck():
        annotated_frame = edge_service.visualize(frame)
        redis_client.rpush(VISUAL_FRAME_QUEUE, pickle.dumps((seq, annotated_frame)))

def poll_queue(redis_client, edge_service):
    """Continuously polls Redis for new frames and enqueues them with sequence number."""
    MAX_REDIS_QUEUE_LEN = 30  # Threshold to drop stale frames
    DROP_BATCH_SIZE = 1       # Number of frames to drop when overloaded

    while True:
        # If local processing queue is full, apply backpressure by not pulling
        if FRAME_QUEUE.full():
            time.sleep(0.01)
            continue

        # If Redis queue is too long, drop old frames to prevent lag
        if redis_client.llen("frame_queue") > MAX_REDIS_QUEUE_LEN:
            for _ in range(DROP_BATCH_SIZE):
                redis_client.lpop("frame_queue")
            logging.warning(f"Dropped {DROP_BATCH_SIZE} old frames from Redis to reduce lag")
            continue

        frame_data = redis_client.blpop("frame_queue", timeout=1)
        if frame_data:
            try:
                frame = pickle.loads(frame_data[1])
                seq = next(frame_counter)
                FRAME_QUEUE.put((seq, frame))
            except Exception as e:
                logging.warning(f"Failed to enqueue frame: {e}")


def frame_worker(edge_service):
    """Consumes frames from queue and processes them using the edge service."""
    while True:
        if not PAUSED.is_set():
            try:
                seq, frame = FRAME_QUEUE.get(timeout=1)
            except Empty:
                continue

            if COOLDOWN.is_set():
                logging.info("Cooldown active. Dropping frame.")
                FRAME_QUEUE.task_done()
                continue

            try:
                process_frame(seq, frame, edge_service)
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
            finally:
                FRAME_QUEUE.task_done()

def clear_thread_queue():
    """Clears all frames from the thread-safe processing queue."""
    while not FRAME_QUEUE.empty():
        try:
            FRAME_QUEUE.get_nowait()
        except Empty:
            break

@app.post("/update-auth-cache")
async def update_auth_cache(payload: dict):
    try:
        edge_service.update_auth_cache(payload)
        return {"status": "cache updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")

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

@app.post("/set-regions")
async def set_regions(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        if data.get("unset"):
            edge_service.unset_region_config()
            return "Region config was unset.", 200
        else:
            edge_service.load_region_config(data)
            return "Region config was loaded.", 200
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing region config: {e}")

@app.get("/healthcheck")
async def healthcheck():
    """Checks if the edge service is running."""
    return "Edge service running!", 200

def main():
    """Initializes services, starts worker threads, and runs the FastAPI app."""
    global redis_client, edge_service
    redis_client = connect_to_redis()
    edge_service = EdgeService("/app/models/yolo11n.pt", "/app/models/license_plate_detector.pt")
    edge_service.on()

    # Start the worker thread and polling thread
    NUM_WORKERS = 4  # Adjust based on your CPU and memory capacity

    for _ in range(NUM_WORKERS):
        threading.Thread(target=frame_worker, args=(edge_service,), daemon=True).start()
    
    threading.Thread(target=poll_queue, args=(redis_client, edge_service), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
