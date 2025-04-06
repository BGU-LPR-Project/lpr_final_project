# Use a base image (adjust according to your setup)
FROM python:3.10-slim

# Install system dependencies including OpenCV requirements, ccache, and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ccache \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the environment variable to use ccache
ENV CCACHE_DISABLE=0
ENV PATH="/usr/lib/ccache:${PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . /app

# Set the working directory
WORKDIR /app

# Run your app
CMD ["python", "app.py"]
