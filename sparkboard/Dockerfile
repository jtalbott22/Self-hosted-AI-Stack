FROM python:3.12-slim

# No docker CLI is installed on purpose. Container and port inventory goes
# through the Docker Engine's REST API over the mounted socket, which needs
# nothing in the image beyond httpx -- already a dependency for the proxy.

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPARKBOARD_BIND=0.0.0.0 \
    SPARKBOARD_PORT=9102 \
    DB_PATH=/data/metrics.db \
    HOST_PROC=/host/proc

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

VOLUME ["/data"]
EXPOSE 9102

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9102/api/health',timeout=4).status==200 else 1)"

# uvicorn is invoked directly rather than through app.server:main so that the
# port can be overridden by compose without the image caring.
CMD ["python", "-m", "uvicorn", "app.server:app", \
     "--host", "0.0.0.0", "--port", "9102", "--log-level", "warning"]
