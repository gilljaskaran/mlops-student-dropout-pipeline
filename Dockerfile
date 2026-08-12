# FastAPI serving image for the student dropout / academic success model.
# Follows the Docker lab conventions (slim base, non-root user, healthcheck)
# and Week 8's "Dockerfile for ML API" pattern.
FROM python:3.9-slim

WORKDIR /app

# curl is needed for the HEALTHCHECK below -- python:3.9-slim doesn't ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so dependency installs are cached across builds
# that only change application code (Docker lab, Task 2.4).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code + the trained model artifact. The model must exist
# locally before building (`dvc repro` or `dvc pull`) -- see
# docs/deployment.md.
COPY app/ ./app/
COPY models/model.pkl ./models/model.pkl

# Non-root user (Docker lab, Task 2.4)
RUN useradd -m -s /bin/bash apiuser && chown -R apiuser:apiuser /app
USER apiuser

ENV MODEL_PATH=models/model.pkl \
    LOG_DIR=/app/logs \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
