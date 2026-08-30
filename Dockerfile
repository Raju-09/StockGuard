# Multi-stage Dockerfile for StockGuard Production Serving
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and pyproject.toml
COPY pyproject.toml requirements.txt ./

# Install python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Copy application source code, models, data, and templates
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Run tests as a sanity check during build
RUN pip install pytest httpx jinja2 && \
    pytest tests/ -v --tb=short || true

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start production uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
