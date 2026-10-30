# SOALM V2 prediction service.
# Lean Linux image serving the Flask API via Waitress.

FROM python:3.12-slim

# System libs some scientific wheels expect at runtime (xgboost needs libgomp).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SOALM_HOST=0.0.0.0 \
    SOALM_PORT=5000 \
    SOALM_THREADS=8

WORKDIR /app

# Install dependencies first so Docker caches this layer when only code changes.
COPY requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt

# Copy the application. .dockerignore keeps out training data and build junk,
# but data/era5/models_v2_final/ and weather/weather.json are included.
COPY . .

EXPOSE 5000

# Container-level health check hits the API's own health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=4).status==200 else 1)"

CMD ["python", "serve.py"]
