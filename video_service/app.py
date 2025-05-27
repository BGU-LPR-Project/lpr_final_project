import time
import redis
import pickle
import uvicorn
import threading
import requests
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from video_handler import VideoHandler

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
    client = redis.StrictRedis(host='redis', port=6379, db=0)
    while True:
        try:
            client.ping()
            break
        except redis.exceptions.BusyLoadingError:
            print("Waiting for Redis to load data into memory...")
            time.sleep(1)
    client.flushall()
    return client

def push_frame_to_redis(frame, redis_client):
    frame_data = pickle.dumps(frame)
    redis_client.rpush("frame_queue", frame_data)

def process_video(video_path_or_stream: str):
    print(f"Starting processing for: {video_path_or_stream}")
    redis_client = connect_to_redis()
    handler = VideoHandler(video_path_or_stream, target_fps=4)
    handler.load_video()
    state["handler"] = handler
    timeout = 30
    last_frame_time = time.time()

    while not state["stop_event"].is_set():
        if state["pause_event"].is_set():
            time.sleep(0.1)
            continue

        frame = handler.decode_frame()
        if frame is None:
            time.sleep(0.1)
            if time.time() - last_frame_time > timeout:
                print("Timeout reached, stopping stream processing.")
                break
            continue

        last_frame_time = time.time()
        push_frame_to_redis(frame, redis_client)

    handler.release_resources()
    print("Finished processing video.")

@app.post("/start-video")
def start_video(input: VideoInput):
    with state["lock"]:
        if state["thread"] and state["thread"].is_alive():
            return {"message": "Video processing already running."}

        state["video_path"] = input.path
        state["pause_event"].clear()
        state["stop_event"].clear()
        requests.get("http://edge_service:8000/edge-resume")

        t = threading.Thread(target=process_video, args=(input.path,))
        t.start()
        state["thread"] = t

    return {"message": f"Started processing {input.path}"}

@app.post("/pause-video")
def pause_video():
    state["pause_event"].set()
    requests.get("http://edge_service:8000/edge-pause")
    return {"message": "Video processing paused."}

@app.post("/resume-video")
def resume_video():
    state["pause_event"].clear()
    requests.get("http://edge_service:8000/edge-resume")
    return {"message": "Video processing resumed."}

@app.post("/stop-video")
def stop_video():
    state["stop_event"].set()
    state["pause_event"].clear()
    requests.get("http://edge_service:8000/clear-queue")
    return {"message": "Video processing stopped."}

@app.post("/restart-video")
def restart_video():
    stop_video()  # stop current processing
    time.sleep(1)  # give thread time to exit
    return start_video(VideoInput(path=state["video_path"]))

@app.post("/skip-10s")
def skip_10_seconds():
    handler = state.get("handler")
    if handler is None:
        return {"message": "No active video handler"}

    try:
        # Call the internal API
        response = requests.get("http://edge_service:8000/skip-at-most-ten", timeout=5)
        if response.status_code == 200:
            data = response.json()
            seconds_skipped = data.get("seconds_skipped", 0)
        else:
            return {"message": "Failed to skip frames"}
    except Exception as e:
        return {"message": f"Error calling skip-at-most-ten: {e}"}

    # Use the result to move the video handler
    handler.seek(10 - seconds_skipped)
    return {"message": "10 seconds skipped."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
