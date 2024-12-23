import cv2
import numpy as np
from typing import List
from ultralytics import YOLO
import torch
from AI1 import BoundingBox
import easyocr

class LicensePlateDetector:

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the LicensePlateDetector class.
        
        Args:
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Minimum confidence required to consider a detection valid.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.reader = easyocr.Reader(['en'], gpu=True)
       
    def detect_license_plates(self, frame: np.ndarray, detected_cars: List[BoundingBox]):
        """
        Detect license plates in the frame and associate them with detected cars.
        Draw bounding boxes for cars and plates, and display OCR results above the car bounding boxes.
        Only consider plates fully contained within a car's bounding box.
        """
        license_plates = self.model(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = license_plate
            if confidence > self.confidence_threshold and int(class_id) == 0:
                plate_box = BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1), confidence)

                # Check if the plate is fully contained within any car bounding box
                for car_box in detected_cars:
                    if (plate_box.x >= car_box.x and
                        plate_box.y >= car_box.y and
                        plate_box.x + plate_box.width <= car_box.x + car_box.width and
                        plate_box.y + plate_box.height <= car_box.y + car_box.height):
                        
                        # Crop the license plate region
                        cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]

                        # Perform OCR on the cropped plate
                        ocr_text = self.read_text_from_plate(cropped_plate)

                        # Draw bounding box for the car
                        cv2.rectangle(frame, (car_box.x, car_box.y), (car_box.x + car_box.width, car_box.y + car_box.height), (255, 0, 0), 2)
                        
                        # Display OCR result above the car bounding box
                        car_label = f"Car: {ocr_text}"
                        # Determine text size
                        (text_width, text_height), baseline = cv2.getTextSize(car_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        # Draw a background rectangle for better readability
                        label_x1 = car_box.x
                        label_y1 = car_box.y - 10 - text_height - baseline
                        label_x2 = car_box.x + text_width
                        label_y2 = car_box.y - 10
                        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (255, 0, 0), -1)

                        # Put text on top of the car bounding box
                        cv2.putText(frame, car_label, (car_box.x, car_box.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Draw bounding box for the license plate
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                        plate_label = f"Conf: {confidence:.2f}"
                        cv2.putText(frame, plate_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)



    def read_text_from_plate(self, cropped_plate: np.ndarray):
        """
        Read text from the cropped license plate using OCR.

        Args:
            cropped_plate (np.ndarray): Cropped image of the license plate.

        Actions:
            - Preprocess the cropped license plate for better OCR results.
            - Use Tesseract OCR to extract text from the image.
            - Display the processed license plate and extracted text.
        """

        # Adjust size based on typical license plate proportions
        license_plate_crop_resized = cv2.resize(cropped_plate.copy(), (300, 100))  
        
        # Preprocess the cropped plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop_resized, cv2.COLOR_BGR2GRAY)
        license_plate_crop_blurred = cv2.GaussianBlur(license_plate_crop_gray, (5, 5), 0)
        license_plate_crop_thres = cv2.adaptiveThreshold(license_plate_crop_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)     

        cv2.imshow("processed plate", license_plate_crop_blurred)   

        # Use OCR model to read license plate
        try:
            return self.reader.readtext(license_plate_crop_thres, detail=0)
            #return pytesseract.image_to_string(license_plate_crop_blurred, lang='eng', config='--psm 6').strip()
            # results = pytesseract.image_to_data(license_plate_crop_gray, lang='eng', config='--psm 6', output_type=pytesseract.Output.DATAFRAME)
            
            # # Filter out weak confidence text localizations
            # filtered_results = results[results['conf'] > 0]

            # for _, row in filtered_results.iterrows():
            #     text = row['text']
            #     conf = row['conf']
                
            #     print(f"Confidence: {conf}")
            #     print(f"Text: {text}")

                # print(f"License Plate detected: {predicted_result}")
        except Exception as e:
            print(f"An error occurred: {e}")