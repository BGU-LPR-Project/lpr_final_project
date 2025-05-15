# Use a minimal base image  
FROM python:3.10-slim  

# Install OS dependencies first (optimized for caching)  
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*  

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
