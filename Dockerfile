# Single-stage lightweight container for CPU/GPU Viscan OPG inference
FROM python:3.11-slim

# Install system dependencies for imaging libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Live CBCT inference is opt-in: the default image runs 2D OPG, pre-segmented
# CBCT label maps and the demo bundles. Build with --build-arg INSTALL_CBCT=true
# and mount the Zenodo weights to enable raw CBCT uploads.
ARG INSTALL_CBCT=false

COPY requirements.txt requirements-cbct.txt ./

RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_CBCT" = "true" ]; then pip install --no-cache-dir -r requirements-cbct.txt; fi

# Copy application source code
COPY . .

# Ensure results and checkpoints directories exist
RUN mkdir -p results checkpoints data

# Expose FastAPI port
EXPOSE 8000

# Set default environment variables
ENV PYTHONUNBUFFERED=1
ENV VISCAN_WEIGHTS_PATH=checkpoints/best_model.pth

# Launch command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
