FROM python:3.10-slim
WORKDIR /app
COPY cloud_main.py .


RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libgomp1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cloud_main.py .

CMD ["python", "cloud_main.py"]
