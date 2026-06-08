# Dockerfile for DrugCLIP Web Application

FROM pytorch/pytorch:2.0.0-cuda11.7-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r web_app/backend/requirements.txt

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=web_app/backend/app.py
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV DEVICE=cuda

# Run the application
CMD ["python", "web_app/backend/app.py"]
