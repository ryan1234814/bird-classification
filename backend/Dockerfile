# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the backend code and the model directory (which is a sibling)
COPY backend /app/backend
COPY model /app/model

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
# We run from the /app directory but point to the backend.main module
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
