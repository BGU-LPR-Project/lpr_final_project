# Redirect to your UI for demo, or any one service for now
FROM python:3.10-slim

WORKDIR /app
COPY ./ui.py .

RUN pip install fastapi uvicorn

CMD ["python", "ui.py"]
