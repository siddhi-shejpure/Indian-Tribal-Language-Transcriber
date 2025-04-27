FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for transcriptions and models
RUN mkdir -p transcriptions models

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MODEL_SIZE=medium \
    OUTPUT_DIR=/app/transcriptions

# Expose port for web interface
EXPOSE 7860

# Set entrypoint
ENTRYPOINT ["python", "mapping.py"]

# Default command (can be overridden)
CMD ["--web", "--host", "0.0.0.0", "--port", "7860"] 