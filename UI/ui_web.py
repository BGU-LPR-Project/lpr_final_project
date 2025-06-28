from flask import Flask, render_template, Response, request, jsonify, abort, redirect, url_for, session
import cv2
import redis
import pickle
import requests
import time
from auth_utils import verify_credentials
import os
import json

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, db=0)
queue_name = 'visual_frame_queue'
current_video_path = "/app/recordings/fullMovie20250420_11.mp4"
last_frame = None

app.secret_key = os.getenv("FLASK_SECRET_KEY", "unsafe-default-dev-key")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = verify_credentials(username, password)
        if role:
            session['username'] = username
            session['role'] = role
            session['logged_in'] = True
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

# Example of protected route
@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.before_request
def require_login():
    allowed_routes = ['login', 'static', 'logout']  # add logout and any other open routes
    if request.endpoint is None:
        return  # Let Flask handle 404 etc.
    if request.endpoint not in allowed_routes and not session.get('logged_in'):
        return redirect(url_for('login'))

def generate():
    global last_frame

    last_seq = -1  # Initialize to -1 so 0 is accepted as the first frame

    while True:
        frame_data = r.lpop(queue_name)
        if frame_data:
            try:
                seq, frame = pickle.loads(frame_data)

                # Only emit if the frame is newer
                if seq > last_seq:
                    last_seq = seq
                    last_frame = frame

                    _, jpeg = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

                # Else: drop older or duplicate frames silently

            except Exception as e:
                print(f"Frame processing error: {e}")
        else:
            time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    # MJPEG video stream
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get-detections")
def get_detections():
    detections = []

    while r.llen('DETECTIONS') > 0:
        item = r.lpop('DETECTIONS')
        if item:
            try:
                detections.append(json.loads(item))
            except Exception as e:
                print("Failed to load JSON:", e)

    return jsonify(detections)

@app.route('/auth-lists', methods=['GET'])
def get_auth_lists():
    """Proxy GET /auth-lists to cloud service."""
    try:
        resp = requests.get(f"http://localhost:8002/auth-lists")
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"detail": "Failed to fetch authorization lists"}), 500


@app.route('/auth-lists/<list_type>', methods=['POST'])
def add_plate(list_type):
    """Proxy POST /auth-lists/{whitelist|blacklist} to cloud service."""
    if list_type not in ['whitelist', 'blacklist']:
        abort(404, description="Invalid list type")

    plate_data = request.get_json()
    if not plate_data or 'plate' not in plate_data:
        abort(400, description="Missing plate in request body")

    try:
        resp = requests.post(
            f"http://localhost:8002/auth-lists/{list_type}",
            json={"plate": plate_data['plate']}
        )
        resp.raise_for_status()
        return jsonify(resp.json()), resp.status_code
    except requests.RequestException as e:
        return jsonify({"detail": f"Failed to add plate to {list_type}"}), 500


@app.route('/auth-lists/<list_type>/<plate>', methods=['DELETE'])
def delete_plate(list_type, plate):
    """Proxy DELETE /auth-lists/{whitelist|blacklist}/{plate} to cloud service."""
    if list_type not in ['whitelist', 'blacklist']:
        abort(404, description="Invalid list type")

    try:
        resp = requests.delete(f"http://localhost:8002/auth-lists/{list_type}/{plate}")
        resp.raise_for_status()
        return jsonify(resp.json()), resp.status_code
    except requests.RequestException as e:
        return jsonify({"detail": f"Failed to delete plate from {list_type}"}), 500

@app.route('/formats/<string:name>', methods=['DELETE'])
def delete_format_proxy(name):
    resp = requests.delete(f'http://localhost:8002/formats/{name}')
    return Response(resp.content, status=resp.status_code, content_type=resp.headers['content-type'])

@app.route('/formats', methods=['GET', 'POST'])
def formats_proxy():
    if request.method == 'GET':
        resp = requests.get(f'http://localhost:8002/formats')
        return Response(resp.content, status=resp.status_code, content_type=resp.headers['content-type'])
    elif request.method == 'POST':
        resp = requests.post(f'http://localhost:8002/formats', json=request.json)
        return Response(resp.content, status=resp.status_code, content_type=resp.headers['content-type'])

@app.route('/set-video-path', methods=['POST'])
def set_video_path():
    global current_video_path
    data = request.json
    path = data.get('path')
    if not path:
        return jsonify({'error': 'No path provided'}), 400
    current_video_path = f"/app/recordings/{path}"
    print(f"Video path updated to: {current_video_path}")
    return jsonify({'message': 'Video path updated'})

@app.route('/list-videos', methods=['GET'])
def list_videos():
    try:
        # Forward the request to your container's API that returns video files list
        response = requests.get("http://localhost:8000/list-videos", timeout=5)
        response.raise_for_status()  # Raise error for bad status
        # Just pass through the JSON response from the container
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': f'Failed to fetch videos from container: {str(e)}'}), 500

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    video = request.files['video']

    # Forward to Docker container's API
    try:
        files = {'video': (video.filename, video.stream, video.mimetype)}
        response = requests.post("http://localhost:8000/save-video", files=files)
        response.raise_for_status()
    except Exception as e:
        return jsonify({'error': f'Failed to send to container: {str(e)}'}), 500

    return jsonify({'message': f'Uploaded and forwarded to container: {video.filename}'})

@app.route('/single_frame')
def single_frame():
    global last_frame
    frame = last_frame
    if frame is not None:
        try:
            _, jpeg = cv2.imencode('.jpg', frame)
            return Response(jpeg.tobytes(), mimetype='image/jpeg')
        except Exception as e:
            return f"Frame encode error: {e}", 500
    return "No frame available", 404


@app.route('/set-regions', methods=['POST'])
def set_regions():
    data = request.json
    print("Received region config:", data)

    # Forward to backend
    try:
        backend_url = "http://localhost:8001/set-regions"
        response = requests.post(backend_url, json=data, timeout=5)
        return response.text, response.status_code
    except Exception as e:
        return f"Failed to forward region config to backend: {e}", 500

@app.route('/start-video', methods=['POST'])
def start_video():
    return proxy_api('/start-video')

@app.route('/pause-video', methods=['POST'])
def pause_video():
    return proxy_api('/pause-video')

@app.route('/resume-video', methods=['POST'])
def resume_video():
    return proxy_api('/resume-video')

@app.route('/restart-video', methods=['POST'])
def restart_video():
    return proxy_api('/restart-video')

@app.route('/skip-10s', methods=['POST'])
def skip_video():
    return proxy_api('/skip-10s')

def proxy_api(endpoint):
    url = f"http://localhost:8000{endpoint}"
    try:
        if endpoint == "/start-video":
            response = requests.post(url, json={"path": current_video_path}, timeout=5)
        else:
            response = requests.post(url, timeout=5)
        return response.text, response.status_code
    except Exception as e:
        return f"Failed to reach backend API: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)