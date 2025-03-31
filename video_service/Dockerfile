FROM python:3.10-slim

# Install OpenCV dependencies clearly
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install opencv-python-headless flask

EXPOSE 8000
CMD ["python", "app.py"]
