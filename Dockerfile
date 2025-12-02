# Base image
FROM python:3.11-slim

# Set work directory inside the container
WORKDIR /workspace

# Install system dependencies (optional but helps with scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Ensure src is on PYTHONPATH (FastAPI app lives in src/api_app.py)
ENV PYTHONPATH=/workspace

# Default envs (can be overridden by HF secrets)
# For local dev or no Firebase, you can set backend to "local"
ENV POSEIDON_STORAGE_BACKEND=firebase
ENV POSEIDON_FIREBASE_CREDENTIALS=/workspace/firebase_service_account.json

# Hugging Face Spaces sets $PORT dynamically. Uvicorn must bind to 0.0.0.0:$PORT
EXPOSE 7860

# Start-up command:
# 1) Write Firebase JSON file from FIREBASE_SERVICE_ACCOUNT_JSON (if provided)
# 2) Start FastAPI with uvicorn
CMD bash -c "python -m src.startup_firebase && uvicorn src.api_app:app --host 0.0.0.0 --port ${PORT:-7860}"
