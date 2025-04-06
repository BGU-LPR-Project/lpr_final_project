import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cloud import CloudService
import threading

# Initialize FastAPI app and CloudService
app = FastAPI()
cloud_service = CloudService()
cloud_service.on()

# Model to structure the input for the /predict endpoint
class PlateImage(BaseModel):
    plate_img: str

# Model to structure the OCR result
class OCRResult(BaseModel):
    ocr_result: tuple

def predict_plate_img(plate_img: np.ndarray):
    """Process the plate image and get prediction asynchronously."""
    result = {"ocr_result": (str(), 0.0)}

    def callback(prediction):
        """Callback to handle the prediction result."""
        nonlocal result
        result["ocr_result"] = prediction

    # Process prediction asynchronously
    cloud_service.predict(plate_img, callback)
    
    return result

@app.post("/predict", response_model=OCRResult)
async def predict(plate_image: PlateImage):
    """Handle prediction requests for license plate number."""
    try:
        plate_img_encoded = plate_image.plate_img

        if not plate_img_encoded:
            raise HTTPException(status_code=400, detail="No plate image provided")

        # Decode the base64 image
        img_bytes = base64.b64decode(plate_img_encoded)
        plate_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Perform prediction
        result = predict_plate_img(plate_img)

        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction")

@app.get("/healthcheck")
async def healthcheck():
    """Health check endpoint to ensure the service is running."""
    return {"status": "Cloud OCR service running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
