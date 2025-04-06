import time
import redis
import pickle
import requests
from video_handler import VideoHandler
import cv2

def wait_for_edge_service():
    """Waits for the edge service to become healthy."""
    while True:
        try:
            response = requests.get("http://edge_service:8000/healthcheck", timeout=5)
            if response.status_code == 200:
                print("Edge service is healthy.")
                break
        except requests.exceptions.RequestException:
            print("Edge service is not yet ready, retrying...")

        time.sleep(1)


def connect_to_redis():
    """Connects to the Redis server and waits for it to be ready."""
    client = redis.StrictRedis(host='redis', port=6379, db=0)
    while True:
        try:
            # Check if Redis is ready to accept commands
            client.ping()
            break
        except redis.exceptions.BusyLoadingError:
            print("Waiting for Redis to load data into memory...")
            time.sleep(1)  # Retry after a short delay
    client.flushall()
    return client


def push_frame_to_redis(frame, redis_client):
    """Serializes and pushes the frame to Redis."""
    while redis_client.llen("frame_queue") >= 30:
        time.sleep(0.1)  # Pause briefly before checking again
    frame_data = pickle.dumps(frame)
    redis_client.rpush("frame_queue", frame_data)


def main(video_path_or_stream):
    """Main function to handle video or RTSP stream processing and Redis interaction."""
    # Wait for edge service and Redis to be ready
    wait_for_edge_service()
    redis_client = connect_to_redis()

    # Open video capture (video file or RTSP stream)
    handler = VideoHandler(video_path_or_stream, target_fps=4)  # Process at 4 FPS
    handler.load_video()

    timeout = 30  # Timeout after 30 seconds of no frames
    last_frame_time = time.time()  # Track when the last frame was receive

    while True:
        # Get a frame from the video or stream
        frame = handler.decode_frame()
    
        if frame is None:
            time.sleep(0.1)
            
            # Optionally, check if the timeout has been reached to break the loop
            if time.time() - last_frame_time > timeout:
                print("Timeout reached, stopping stream processing.")
                break
            
            continue  # Skip if no frame available

        # Frame received, reset the timeout counter
        last_frame_time = time.time()
        
        # frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)

        # Process the frame (e.g., push to Redis, detect objects, etc.)
        push_frame_to_redis(frame, redis_client)

    handler.release_resources()


if __name__ == "__main__":
    video_path_or_stream = "/app/recordings/motion4.mp4"
    main(video_path_or_stream)
