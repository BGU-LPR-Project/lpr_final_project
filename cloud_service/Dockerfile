FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install flask paddleocr numpy opencv-python-headless ultralytics scipy matplotlib scikit-image

EXPOSE 8000
CMD ["python", "app.py"]
