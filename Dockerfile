# Multi-stage build for smaller image
# Use CUDA-enabled base image for GPU support (can also run on CPU)
FROM nvidia/cuda:12.9.2-devel-ubuntu22.04 AS builder

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage - CUDA runtime for GPU support
FROM nvidia/cuda:12.9.2-runtime-ubuntu22.04

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY main.py audio_streamer.py transcriber.py connection_manager.py transcript_storage.py ./
COPY templates/ ./templates/

# Create directory for transcripts
RUN mkdir -p /app/transcripts

# Expose port
EXPOSE 8000

# Health check - give more time for model download on first run
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/status')" || exit 1

# Run the application directly with uvicorn
# Using 0.0.0.0 explicitly and workers=1 for the transcription worker to function correctly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--access-log"]
