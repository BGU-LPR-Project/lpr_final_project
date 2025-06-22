import base64
import cv2
import numpy as np
from typing import Dict
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from cloud import CloudService
import requests

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

class PlateRequest(BaseModel):
    plate: str

# Model for adding a new format
class FormatAddRequest(BaseModel):
    name: str
    pattern: str

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

def notify_edge_service_update():
    try:
        auth_lists = cloud_service.get_authorization_lists()  # your dict payload
        response = requests.post(
            'http://edge_service:8000/update-auth-cache',
            json=auth_lists  # send the dict as JSON payload
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[WARNING] Failed to notify edge service to update auth cache: {e}")

@app.get("/auth-lists")
async def get_auth_lists():
    return cloud_service.get_authorization_lists()

@app.get("/auth-status")
async def get_auth_status(plate: str = Query(..., description="License plate to check")):
    status = cloud_service.check_plate_authorization(plate)
    return {"plate": plate, "status": status}

@app.post("/auth-lists/whitelist")
async def add_whitelist(plate_req: PlateRequest):
    success = cloud_service.add_to_whitelist(plate_req.plate)
    if not success:
        raise HTTPException(status_code=400, detail="Plate already in whitelist")
    notify_edge_service_update()
    return {"message": f"Plate {plate_req.plate} added to whitelist."}

@app.post("/auth-lists/blacklist")
async def add_blacklist(plate_req: PlateRequest):
    success = cloud_service.add_to_blacklist(plate_req.plate)
    if not success:
        raise HTTPException(status_code=400, detail="Plate already in blacklist")
    notify_edge_service_update()
    return {"message": f"Plate {plate_req.plate} added to blacklist."}

@app.delete("/auth-lists/whitelist/{plate}")
async def remove_whitelist(plate: str):
    success = cloud_service.remove_from_whitelist(plate)
    if not success:
        raise HTTPException(status_code=404, detail="Plate not found in whitelist")
    notify_edge_service_update()
    return {"message": f"Plate {plate} removed from whitelist."}

@app.delete("/auth-lists/blacklist/{plate}")
async def remove_blacklist(plate: str):
    success = cloud_service.remove_from_blacklist(plate)
    if not success:
        raise HTTPException(status_code=404, detail="Plate not found in blacklist")
    notify_edge_service_update()
    return {"message": f"Plate {plate} removed from blacklist."}

@app.get("/formats", response_model=Dict[str, str])
async def get_formats():
    """
    Retrieve all license plate formats from backend.
    Returns a dict of format_name -> regex_pattern.
    """
    try:
        formats = cloud_service.get_formats()  # Expected dict[str, str]
        return formats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch formats: {e}")

@app.post("/formats")
async def add_format(format_req: FormatAddRequest):
    """
    Add a new license plate format.
    """
    try:
        success = cloud_service.add_format(format_req.name, format_req.pattern)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to add format (duplicate or invalid).")
        return {"message": f"Format '{format_req.name}' added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add format: {e}")

@app.delete("/formats/{name}")
async def delete_format(name: str):
    """
    Delete a license plate format by name.
    """
    try:
        success = cloud_service.delete_format(name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Format '{name}' not found.")
        return {"message": f"Format '{name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete format: {e}")

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
