# Multi-stage build: slim runtime, no build toolchain in the final image.
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build deps for pymupdf / numpy wheels (only used in builder stage).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


FROM python:3.13-slim AS runtime

# Run as non-root.
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app

# Bring in installed packages.
COPY --from=builder /root/.local /home/app/.local
ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/home/app/app

# Source — exclude PDFs (only needed at ingestion, not at request time) and
# the local cache directory (volume-mounted in dev, DynamoDB in prod).
COPY --chown=app:app app.py /home/app/app/app.py
COPY --chown=app:app Services /home/app/app/Services
COPY --chown=app:app static /home/app/app/static

USER app
WORKDIR /home/app/app

EXPOSE 8000

# Healthcheck used by App Runner / docker-compose / local docker run.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"

# --proxy-headers + --forwarded-allow-ips='*' so slowapi's get_remote_address
# sees the real client IP via X-Forwarded-For (App Runner sits in front).
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
