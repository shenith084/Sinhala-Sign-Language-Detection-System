# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# Hugging Face spaces use port 7860 by default
ENV PORT 7860
EXPOSE 7860

# Set up the working directory
WORKDIR /app

# Install system dependencies (OpenCV requires libgl)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements and install them
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . ./

# Change to backend directory so imports like 'routes' work correctly
WORKDIR /app/backend

# Run the web service on container startup using gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
