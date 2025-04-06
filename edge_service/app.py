import base64
import cv2
import redis
import threading
import time
import requests
import pickle
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from edge import EdgeService
import logging

# Constants
THREAD_POOL_SIZE = 1  # Adjust as needed based on the number of concurrent requests you want to handle

# FastAPI app initialization
app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def wait_for_cloud_service():
    """Wait for the cloud service to become healthy."""
    while True:
        try:
            response = requests.get("http://cloud_service:8000/healthcheck", timeout=5)
            if response.status_code == 200:
                print("Cloud service is healthy.")
                break
        except requests.exceptions.RequestException:
            print("Cloud service is not yet ready, retrying...")

        time.sleep(1)


def connect_to_redis():
    """Connect to the Redis server and wait for it to be ready."""
    client = redis.StrictRedis(host='redis', port=6379, db=0)
    while True:
        try:
            # Check if Redis is ready to accept commands
            client.ping()
            break
        except redis.exceptions.BusyLoadingError:
            print("Waiting for Redis to load data into memory...")
            time.sleep(1)  # Retry after a short delay
    return client

def process_frame(frame, edge_service):
    """Process each frame using edge service and send results to cloud."""
    print("Processing frame in new thread.")

    result = {}

    def callback(prediction):
        nonlocal result
        result = prediction

    # Start async prediction
    edge_service.predict(frame, callback)

    # Handle results now that callback is done
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
        except Exception as e:
            print(f"cloud predict api failed: {e}")


def poll_queue(redis_client, executor, edge_service):
    """Poll the Redis queue and process frames asynchronously."""
    while True:
        # logging.info(redis_client.llen("frame_queue"))
        frame_data = redis_client.lpop("frame_queue")  # Block until a frame is available
        if frame_data:
            frame = pickle.loads(frame_data)  # Deserialize the frame

            # Use thread pool to process the frame
            process_frame(frame, edge_service)
        edge_service.log_results()
        time.sleep(0.1)  # Periodic check if needed

def start_polling():
    wait_for_cloud_service()

    # Connect to Redis and initialize services
    redis_client = connect_to_redis()
    edge_service = EdgeService("/app/models/yolo11n.pt", "/app/models/license_plate_detector.pt")
    edge_service.on()

    # Thread pool initialization
    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

    # Start polling loop in a separate thread
    threading.Thread(target=poll_queue, args=(redis_client, executor, edge_service), daemon=True).start()

    return {"status": "Polling started"}

# FastAPI route to health check
@app.get("/healthcheck")
async def healthcheck():
    return "Edge service running!", 200

# Main function to run the FastAPI app
def main():
    wait_for_cloud_service()

    # Connect to Redis and initialize services
    redis_client = connect_to_redis()
    edge_service = EdgeService("/app/models/yolo11n.pt", "/app/models/license_plate_detector.pt")
    edge_service.on()

    # Thread pool initialization
    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

    # Start polling loop in a separate thread
    threading.Thread(target=poll_queue, args=(redis_client, executor, edge_service), daemon=True).start()

    # Run FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
