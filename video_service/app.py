import cv2
import redis
import pickle
import time
import requests
from video_handler import VideoHandler


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
    
    return client


def push_frame_to_redis(frame, redis_client):
    """Serializes and pushes the frame to Redis."""
    while redis_client.llen("frame_queue") >= 30:
        time.sleep(0.1)  # Pause briefly before checking again
    frame_data = pickle.dumps(frame)
    redis_client.rpush("frame_queue", frame_data)
    print(f"Frame pushed to queue: {time.time()}")


def main():
    """Main function to handle video processing and Redis interaction."""
    # Wait for edge service and Redis to be ready
    wait_for_edge_service()
    redis_client = connect_to_redis()

    # Path to video
    video_path = "/app/recordings/motion4.mp4"

    # Open video capture
    handler = VideoHandler(video_path)
    handler.load_video()

    # Variable to track the last time a frame was pushed
    last_frame_time = time.time()
    frame_interval = 1 / 4  # 4 FPS -> 250ms per frame

    while True:
        frame = handler.decode_frame()
        if frame is None:
            break

        current_time = time.time()
        time_since_last_frame = current_time - last_frame_time

        # If 250ms has passed since the last frame, push the frame to Redis
        if time_since_last_frame >= frame_interval:
            push_frame_to_redis(frame, redis_client)

            # Update the last frame push time
            last_frame_time = current_time
        else:
            # Sleep for the remaining time before the next frame can be pushed
            time.sleep(frame_interval - time_since_last_frame)

    handler.release_resources()


if __name__ == "__main__":
    main()
