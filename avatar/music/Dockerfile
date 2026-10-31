FROM python:3.12-slim
# ffmpeg repairs YouTube's DASH audio files so iPhones can play and seek them.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
# yt-dlp needs a JavaScript runtime (deno) to read YouTube's player; the deno
# wheel ships the binary.
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx "yt-dlp[default]" deno
WORKDIR /app
COPY music.py .
ENV CACHE_DIR=/cache CACHE_GB=2
EXPOSE 8000
# yt-dlp changes often to keep up with YouTube. If songs stop playing:
#   docker compose build --no-cache music && docker compose up -d music
CMD ["uvicorn", "music:app", "--host", "0.0.0.0", "--port", "8000"]
