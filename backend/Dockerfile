# Production image for the nutrisnap API (used by Render; also runs anywhere Docker does).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# CLIP ONNX weights (~345 MB: int8 vision + fp32 text tower) are fetched at build time, not committed.
# The fp32 text tower only runs here, to pre-build the cached label embeddings; it is not loaded per request.
RUN python -m scripts.download_models \
 && python -c "from app import recognize; recognize.load_model()"

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

ENV ENV=production
EXPOSE 8000
# Render sets $PORT. Proxy headers are trusted so rate limiting sees the client IP.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
