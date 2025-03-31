FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies first (clearly optimized for caching)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies clearly with caching optimization
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy your code only AFTER dependencies installed clearly
COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
