import time
import redis
import pickle
import uvicorn
import threading
import requests
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from video_handler import VideoHandler
import os
import shutil

app = FastAPI()

class VideoInput(BaseModel):
    path: str

# State management
state = {
    "thread": None,
    "video_path": None,
    "pause_event": threading.Event(),
    "stop_event": threading.Event(),
    "lock": threading.Lock(),
    "handler": None,
}

def connect_to_redis():
    """Establishes a connection to Redis, flushes keys, and waits if loading."""
    client = redis.StrictRedis(host='redis', port=6379, db=0)
    while True:
        try:
            client.ping()  # Try to ping Redis
            break
        except redis.exceptions.BusyLoadingError:
            print("Waiting for Redis to load data into memory...")
            time.sleep(1)
    client.flushall()  # Clear all existing keys
    return client

def push_frame_to_redis(frame, redis_client):
    """Serializes and pushes a video frame to the Redis queue."""
    frame_data = pickle.dumps(frame)  # Convert frame to bytes
    redis_client.rpush("frame_queue", frame_data)  # Push to Redis list

def process_video(video_path_or_stream: str):
    """Processes video frames from a file/stream and pushes them to Redis."""
    print(f"Starting processing for: {video_path_or_stream}")
    redis_client = connect_to_redis()
    handler = VideoHandler(video_path_or_stream, target_fps=4)
    handler.load_video()
    state["handler"] = handler

    timeout = 30  # Max wait time for frames
    last_frame_time = time.time()

    while not state["stop_event"].is_set():
        if state["pause_event"].is_set():
            time.sleep(0.1)  # Wait while paused
            continue

        frame = handler.decode_frame()
        if frame is None:
            if time.time() - last_frame_time > timeout:
                print("Timeout reached, stopping stream processing.")
                break
            time.sleep(0.001)  # Prevent tight spinning
            continue

        last_frame_time = time.time()
        push_frame_to_redis(frame, redis_client)

        # Sleep to match target FPS
        time.sleep(1 / handler.target_fps)

    handler.release_resources()
    print("Finished processing video.")


@app.post("/start-video")
def start_video(input: VideoInput):
    """Stops any running video and starts new processing in a fresh thread."""
    with state["lock"]:
        # If video is running, signal it to stop
        if state["thread"] and state["thread"].is_alive():
            stop_video()
            state["thread"].join()
            state["thread"] = None
            print("Previous video processing stopped.")

        # Start new video
        state["video_path"] = input.path
        state["pause_event"].clear()
        state["stop_event"].clear()
        requests.get("http://edge_service:8000/edge-resume")  # Resume edge service

        t = threading.Thread(target=process_video, args=(input.path,))
        t.start()
        state["thread"] = t

    return {"message": f"Started processing {input.path}"}


@app.post("/pause-video")
def pause_video():
    """Pauses the currently running video processing."""
    state["pause_event"].set()
    requests.get("http://edge_service:8000/edge-pause")  # Pause edge service
    return {"message": "Video processing paused."}

@app.post("/resume-video")
def resume_video():
    """Resumes the paused video processing."""
    state["pause_event"].clear()
    requests.get("http://edge_service:8000/edge-resume")  # Resume edge service
    return {"message": "Video processing resumed."}

@app.post("/stop-video")
def stop_video():
    """Stops the video processing and clears the Redis queue."""
    state["stop_event"].set()
    state["pause_event"].clear()
    requests.get("http://edge_service:8000/clear-queue")  # Clear frame queue
    return {"message": "Video processing stopped."}

@app.post("/restart-video")
def restart_video():
    """Restarts the video processing using the last video path."""
    stop_video()
    time.sleep(1)  # Give thread time to exit cleanly
    return start_video(VideoInput(path=state["video_path"]))

@app.post("/skip-10s")
def skip_10_seconds():
    """Skips up to 10 seconds of video by coordinating with the edge service."""
    handler = state.get("handler")
    if handler is None:
        return {"message": "No active video handler"}

    try:
        # Ask edge-service to drop at most 40 frames (10 seconds)
        response = requests.get("http://edge_service:8000/skip-at-most-ten", timeout=5)
        response.raise_for_status()
        data = response.json()
        seconds_skipped = data.get("seconds_skipped")

        if seconds_skipped is None:
            return {"message": "Invalid response from edge-service"}
    except Exception as e:
        return {"message": f"Error calling edge-service: {e}"}

    # Handle local skip (e.g., when playing back a saved video)
    try:
        handler.seek(10 - seconds_skipped)
    except Exception as e:
        return {"message": f"Error seeking video: {e}"}

    return {"message": f"Skipped ~10s (edge skipped {seconds_skipped}s, local skipped {10 - seconds_skipped}s)"}


@app.post("/save-video")
async def save_video(video: UploadFile = File(...)):
    SAVE_DIR = "recordings"
    os.makedirs(SAVE_DIR, exist_ok=True)
    try:
        file_location = os.path.join(SAVE_DIR, video.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        return f"Saved {video.filename} in container."
    except Exception as e:
        return f"Error saving file: {e}"

@app.get("/list-videos")
async def list_videos():
    files = os.listdir('recordings')  # adjust your folder path
    video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return video_files  # FastAPI converts list to JSON automatically


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
