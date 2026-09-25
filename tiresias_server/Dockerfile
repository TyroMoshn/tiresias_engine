# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

# System runtime dependencies: libgomp1 for FAISS OpenMP, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Non-root user setup
RUN groupadd -g 1000 tiresias && \
    useradd -u 1000 -g tiresias -s /bin/bash -m tiresias && \
    mkdir -p /app/data /app/db && \
    chown -R tiresias:tiresias /app

# Install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code
COPY --chown=tiresias:tiresias app /app/app

# Default environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    TIRESIAS_ROOT="/app" \
    TIRESIAS_DATA_ROOT="/app/data" \
    TIRESIAS_DB_PATH="/app/db/tiresias_user.db" \
    TIRESIAS_PROFILE="performance" \
    TIRESIAS_HOST="0.0.0.0" \
    TIRESIAS_PORT=8000

USER tiresias

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/system/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
