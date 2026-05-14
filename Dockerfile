# Production image for the MAGIC Gamma Telescope inference API.
#
# Build:   docker build -t magic-gamma:v1 .
# Run:     docker run --rm -p 8000:8000 magic-gamma:v1
# Test:    curl http://localhost:8000/health
#
# Image is multi-stage to keep the runtime layer small. The model artifact
# (~few MB) is copied in at build time so the container is fully self-contained.

FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps for any packages with native extensions (lightgbm, xgboost)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user --upgrade pip \
 && pip install --no-cache-dir --user -r requirements.txt \
 # Add deployment-only deps not in the training requirements
 && pip install --no-cache-dir --user fastapi 'uvicorn[standard]' joblib


FROM python:3.12-slim AS runtime

# libgomp1 is needed at runtime by lightgbm / xgboost
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Non-root user so the container doesn't run as root
RUN useradd --create-home --shell /bin/bash magic
USER magic
WORKDIR /home/magic/app

# Pull the installed Python packages from the builder stage
COPY --from=builder --chown=magic:magic /root/.local /home/magic/.local
ENV PATH=/home/magic/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# App code + model artifact
COPY --chown=magic:magic src/        ./src/
COPY --chown=magic:magic server.py   ./
COPY --chown=magic:magic artifacts/  ./artifacts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
