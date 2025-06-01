import base64
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cloud import CloudService

# Initialize FastAPI app and cloud-based OCR service
app = FastAPI()
cloud_service = CloudService()
cloud_service.on()

# Input data model for /predict endpoint
class PlateImage(BaseModel):
    plate_img: str  # Base64-encoded image of the license plate

# Output data model for OCR results
class OCRResult(BaseModel):
    ocr_result: tuple  # Tuple containing (recognized_text, confidence)

def predict_plate_img(plate_img: np.ndarray):
    """
    Run OCR prediction on the plate image using the CloudService.

    Args:
        plate_img (np.ndarray): Decoded license plate image.

    Returns:
        dict: OCR result with 'ocr_result' as (text, confidence).
    """
    result = {"ocr_result": (str(), 0.0)}

    def callback(prediction):
        """Capture prediction result via callback."""
        nonlocal result
        result["ocr_result"] = prediction

    # Perform asynchronous prediction
    cloud_service.predict(plate_img, callback)
    return result

@app.post("/predict", response_model=OCRResult)
async def predict(plate_image: PlateImage):
    """
    Predict license plate number from base64-encoded image.

    Args:
        plate_image (PlateImage): Request body with base64 plate image.

    Returns:
        OCRResult: Recognized text and confidence.
    """
    try:
        plate_img_encoded = plate_image.plate_img

        if not plate_img_encoded:
            raise HTTPException(status_code=400, detail="No plate image provided")

        # Decode base64 string to image
        img_bytes = base64.b64decode(plate_img_encoded)
        plate_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Perform OCR prediction
        result = predict_plate_img(plate_img)
        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction")

@app.get("/healthcheck")
async def healthcheck():
    """
    Endpoint to verify service health.

    Returns:
        dict: Health status of the OCR service.
    """
    return {"status": "Cloud OCR service running!"}

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
