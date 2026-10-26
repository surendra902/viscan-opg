# CPU image for the 2D OPG service. Build with --build-arg INSTALL_CBCT=true
# to add the live 3D stack; GPU deployments should start from a CUDA base image.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG INSTALL_CBCT=false

COPY requirements.txt requirements-cbct.txt ./
# The default PyPI torch pulls CUDA libraries this slim image cannot use.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_CBCT" = "true" ]; then pip install --no-cache-dir -r requirements-cbct.txt; fi

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p results checkpoints data \
    && chown -R app:app /app
USER app

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV VISCAN_WEIGHTS_PATH=checkpoints/best_model.pth

# python app.py, not bare uvicorn: the first start fetches the checkpoint and
# verifies its SHA-256 before serving. uvicorn app:app skips that bootstrap.
CMD ["python", "app.py"]
