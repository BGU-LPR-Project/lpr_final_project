from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from cloud import CloudService

# Initialize Flask app and CloudService
app = Flask(__name__)
cloud_service = CloudService()
cloud_service.on()

def process_prediction(plate_img, callback):
    """Process the plate image and handle prediction callback."""
    try:
        cloud_service.predict(plate_img, callback)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests for license plate number."""
    try:
        data = request.json
        plate_img_encoded = data.get("plate_img")

        if not plate_img_encoded:
            return jsonify({"error": "No plate image provided"}), 400

        # Decode the base64 image
        img_bytes = base64.b64decode(plate_img_encoded)
        plate_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Result variable to hold the prediction
        result = {"plate_number": "---"}

        def callback(prediction):
            """Callback to handle the prediction result."""
            result["plate_number"] = prediction

        # Process prediction asynchronously
        process_prediction(plate_img, callback)

        return jsonify(result)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Health check endpoint to ensure the service is running."""
    return "Cloud OCR service running!", 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
