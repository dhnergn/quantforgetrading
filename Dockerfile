# Multi-stage build for cross-platform deployment
FROM python:3.12-slim as base

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY quantforge/ ./quantforge/
COPY scripts/ ./scripts/
COPY config.yaml .
COPY setup.py .

# Install package
RUN pip install -e .

# Create non-root user for security
RUN useradd -m quantforge && chown -R quantforge:quantforge /app
USER quantforge

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "quantforge.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
