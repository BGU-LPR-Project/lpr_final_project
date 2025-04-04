# Use a minimal base image  
FROM python:3.10-slim  

# Set working directory  
WORKDIR /app  

# Install OS dependencies first (optimized for caching)  
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*  

# Install Python dependencies separately to leverage Docker cache  
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  

# Copy only relevant files after dependencies are installed  
COPY . .  

# Expose the required port  
EXPOSE 8000  

# Use a more explicit command for better readability  
CMD ["python3", "app.py"]
