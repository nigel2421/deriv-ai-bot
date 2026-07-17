# Cloud Run + local container image for Deriv AI Bot
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cloud-oriented requirements (Linux; no Windows-only pins)
COPY requirements-cloud.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY .env.example ./.env.example

RUN mkdir -p /app/data/logs /app/data/historical /app/data/training /app/src/models

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.cloud_app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 75"]
