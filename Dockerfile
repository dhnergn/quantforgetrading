# QuantForge AI
#
# The image runs the API with live trading disabled, which is the only mode
# this container can start in. The live-trading gate additionally requires a
# signed authorisation record on a mounted volume, so the container cannot
# reach a live venue on its own.

FROM python:3.12-slim

# Pinned, and installed before the source is copied so a source change does
# not invalidate the dependency layer.
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY quantforge/ ./quantforge/
COPY scripts/       ./scripts/
COPY tests/         ./tests/
COPY config.yaml setup.py README.md ./

# Non-root: the container runs no code that needs privilege.
RUN useradd --create-home --uid 1000 quantforge \
 && mkdir -p /app/.quantforge /app/logs \
 && chown -R quantforge:quantforge /app
USER quantforge

# Loopback by default. Binding 0.0.0.0 is a deliberate act, not a default.
ENV QUANTFORGE_ALLOW_STALE=0
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "quantforge.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
