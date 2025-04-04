FROM python:3.10-slim
WORKDIR /app
COPY edge_main.py .

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY edge_main.py .

CMD ["python", "edge_main.py"]
