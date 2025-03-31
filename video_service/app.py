from flask import Flask, Response
from video_handler import VideoHandler
import cv2

app = Flask(__name__)

video_handler = VideoHandler("/app/recordings/motion4.mp4")
video_handler.load_video()



@app.route('/frame', methods=['GET'])
def get_frame():
    frame = video_handler.decode_frame()
    if frame is None:
        return Response(status=204)  # No content
    _, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return "Video service running!", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
