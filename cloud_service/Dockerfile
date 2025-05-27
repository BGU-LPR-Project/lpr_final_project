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

# Set working directory  
WORKDIR /app  

# Copy only necessary files first  
COPY requirements.txt .  

# Install Python dependencies (optimized for caching)  
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy the rest of the application files  
COPY . .  

# Expose the required port  
EXPOSE 8000  

# Use an explicit Python 3 command  
CMD ["python3", "app.py"]
