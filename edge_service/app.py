import base64
import cv2
import numpy as np
import redis
import threading
import time
import requests
import pickle
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify
from edge import EdgeService
from auth_manager import AuthManager

# Constants
THREAD_POOL_SIZE = 4  # Adjust as needed based on the number of concurrent requests you want to handle


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


def process_frame(frame, edge_service, result):
    """Process each frame using edge service and send results to cloud."""
    print("Processing frame in new thread.")

    def callback(prediction):
        nonlocal result
        result = prediction
        #print("Edge prediction callback executed with result:", result)

    # Process with edge service
    edge_service.predict(frame, callback)

    # Handle results
    for object_id, data in result.items():
        plate_img = data.get("plate_bbox")
        authorization = 0  # Default unauthorized

        if plate_img:
            # Send plate image to cloud service
            _, buffer = cv2.imencode('.jpg', frame[plate_img[1]:plate_img[3], plate_img[0]:plate_img[2]])
            encoded_plate = base64.b64encode(buffer).decode('utf-8')

            cloud_response = requests.post(
                "http://cloud_service:8000/predict", json={"plate_img": encoded_plate}, verify=False).json()
            plate_number = cloud_response.get("plate_number", "---")

            print(plate_number)


def poll_queue(redis_client, executor, edge_service):
    """Poll the Redis queue and process frames asynchronously."""
    while True:
        frame_data = redis_client.lpop("frame_queue")  # Block until a frame is available
        if frame_data:
            frame = pickle.loads(frame_data)  # Deserialize the frame
            
            # Use thread pool to process the frame
            result = {}
            executor.submit(process_frame, frame, edge_service, result)
        time.sleep(0.1)  # Periodic check if needed


def create_flask_app():
    """Create and configure Flask application."""
    app = Flask(__name__)

    @app.route('/healthcheck', methods=['GET'])
    def healthcheck():
        return "Edge service running!", 200

    return app


def main():
    """Main function to start the services and the polling loop."""
    wait_for_cloud_service()
    
    # Connect to Redis and initialize services
    redis_client = connect_to_redis()
    edge_service = EdgeService("/app/models/yolo11n.pt", "/app/models/license_plate_detector.pt")
    edge_service.on()
    auth_manager = AuthManager(['NA13NRU'], ['GX15OGJ'])

    # Thread pool initialization
    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

    # Start polling loop in a separate thread
    threading.Thread(target=poll_queue, args=(redis_client, executor, edge_service), daemon=True).start()

    # Start Flask app
    app = create_flask_app()
    app.run(host='0.0.0.0', port=8000)


if __name__ == "__main__":
    main()
