# ApicalAI — Distributed AI Inference Engine
# Multi-stage Docker build
# ============================================

FROM python:3.11-slim AS base

LABEL maintainer="ApicalAI Team"
LABEL description="Distributed AI Inference Engine — from OS primitives to agentic AI"
LABEL version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (only numpy)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Expose API port
EXPOSE 8080

# Default: run the API server
ENV PYTHONPATH=/app
CMD ["python", "-m", "api.app"]
