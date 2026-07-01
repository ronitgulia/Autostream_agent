# Multi-stage production Dockerfile for AutoStream AI Agent
FROM python:3.11-slim as base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies (build-essential needed for some C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies into system environment
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root application user for defense-in-depth security
RUN useradd -m -u 10001 -s /bin/bash appuser

# Copy application source code
COPY . .

# Ensure appuser owns runtime data directories
RUN mkdir -p /app/.chroma_db && \
    touch /app/leads.csv && \
    chown -R appuser:appuser /app

USER appuser

# Expose FastAPI webhook port
EXPOSE 8000

# Container healthcheck querying the /healthz endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command starts the FastAPI webhook server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
