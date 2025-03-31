import base64

from flask import Flask, jsonify
import cv2
import numpy as np
import requests
from edge import EdgeService
from auth_manager import AuthManager  # Clearly keep your authorization logic here

app = Flask(__name__)

edge_service = EdgeService("/app/models/yolo11n.pt", "/app/models/license_plate_detector.pt")
edge_service.on()

auth_manager = AuthManager(['NA13NRU'], ['GX15OGJ'])


@app.route('/process_frame', methods=['GET'])
def process_frame():
    print("Flask /process_frame route clearly hit!", flush=True)
    frame_response = requests.get("http://video_service:8000/frame")
    if frame_response.status_code != 200:
        return jsonify({"error": "No frame received"}), 204

    frame = cv2.imdecode(np.frombuffer(frame_response.content, np.uint8), cv2.IMREAD_COLOR)

    result = {}

    def callback(prediction):
        nonlocal result
        result = prediction
        print("Edge prediction callback clearly executed with result:", result, flush=True)

    print("Calling edge_service.predict now clearly.", flush=True)

    edge_service.predict(frame, callback)

    print("Completed edge_service.predict clearly.", flush=True)

    # Visualization & Authorization logic clearly restored from main.py
    for object_id, data in result.items():
        plate_img = data.get("plate_bbox")
        authorization = 0  # default unauthorized

        if plate_img:
            # Send plate image to cloud service clearly
            _, buffer = cv2.imencode('.jpg', frame[plate_img[1]:plate_img[3], plate_img[0]:plate_img[2]])
            encoded_plate = base64.b64encode(buffer).decode('utf-8')

            cloud_response = requests.post("http://cloud_service:8000/predict",
                                           json={"plate_img": encoded_plate}).json()
            plate_number = cloud_response.get("plate_number", "---")

            # Perform authorization clearly
            authorization = auth_manager.get_vehicle_authorization(plate_number)
            data["plate_number"] = plate_number
            data["authorization"] = authorization
        else:
            data["plate_number"] = "---"
            data["authorization"] = authorization

        # Visualization (drawing boxes, labels, authorization clearly)
        visualize(frame, object_id, data, authorization)

    # Return the processed image with visualization clearly
    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes(), 200, {'Content-Type': 'image/jpeg'}


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return "Edge service running!", 200


def visualize(frame, object_id, data, authorized):
    print(f"Visualizing object_id: {object_id}, data: {data}, authorized: {authorized}")

    centroid = data["centroid"]
    bbox = data["bbox"]
    plate_number = data["plate_number"]
    direction = data.get("direction", "unknown")

    box_color = (0, 255, 0) if authorized == 1 else ((0, 0, 255) if authorized == -1 else (0, 0, 255))

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.circle(frame, tuple(centroid), 5, box_color, -1)
    text = f"ID {object_id} - {direction}"
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    if plate_number != "---":
        cv2.putText(frame, f"Plate: {plate_number}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
