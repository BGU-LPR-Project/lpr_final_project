from flask import Flask, request, jsonify
import cv2
import numpy as np
from cloud import CloudService
import base64

app = Flask(__name__)
cloud_service = CloudService()
cloud_service.on()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    plate_img_encoded = data["plate_img"]
    img_bytes = base64.b64decode(plate_img_encoded)
    plate_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    result = {}
    def callback(prediction):
        nonlocal result
        result = prediction

    cloud_service.predict(plate_img, callback)

    return jsonify({"plate_number": result})

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return "Cloud OCR service running!", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
