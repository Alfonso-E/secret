# Slim Python image — runs the bot in any container host (Hetzner, Fly, Cloud Run, ...).
# Build:  docker build -t crypto-bot .
# Run:    docker run --rm --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs crypto-bot
FROM python:3.11-slim

# OS-level deps that lightgbm needs at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*

# Use UTC inside the container so timestamps line up with Bitget's clock
ENV TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first so the layer caches across code changes
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Persistent data + logs live OUTSIDE the image (mount as volumes)
RUN mkdir -p /app/data /app/logs
VOLUME ["/app/data", "/app/logs"]

# Healthcheck: heartbeat file must have been touched in the last 2 hours
HEALTHCHECK --interval=15m --timeout=10s --start-period=2m --retries=3 \
  CMD python -c "import time, pathlib, sys; \
hb=pathlib.Path('/app/logs/heartbeat'); \
sys.exit(0 if hb.exists() and (time.time()-hb.stat().st_mtime) < 7200 else 1)"

# Default: continuous loop in DRY-RUN. Override with --live in `docker run`
# when you're ready for real demo trading.
CMD ["python", "live_bot.py", "--loop"]
