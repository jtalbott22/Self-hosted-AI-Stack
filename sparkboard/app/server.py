"""
Sparkboard -- system and GPU telemetry for NVIDIA DGX Spark.

Runs a background sampler thread that writes to SQLite, and serves:

  GET  /                 the dashboard
  GET  /api/info         static host/GPU facts (queried once at startup)
  GET  /api/now          most recent sample
  GET  /api/live         in-memory ring buffer, for instant chart population
  GET  /api/history      bucketed time series: ?range=1h&gpu=0
  GET  /api/stream       server-sent events, one message per sample
  GET  /api/health       liveness probe
  GET  /metrics          Prometheus text exposition

Everything is served relative to the mount point, so the app works unchanged
whether it's at the domain root or behind an nginx subpath like /gpu/.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from . import collector, config, services
from .proxy import ClassifyingProxy, ProxyConfig, PromptFeed
from .store import Store

logging.basicConfig(
    level=config.LOGLEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("sparkboard")
# httpx logs a line per request at INFO. The Docker inventory makes one call
# per container every few seconds, so left alone it buries the service's own
# log in traffic that says nothing.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

INTERVAL = config.INTERVAL
# Containers and ports change on a human timescale, and a stats sweep costs a
# second or two, so this runs well off the telemetry cadence.
SERVICES_INTERVAL = config.SERVICES_INTERVAL
# The classifying proxy is opt-in. When off, none of its routes are mounted
# and the dashboard simply doesn't show the activity feed.
PROXY_ENABLED = config.PROXY_ENABLED
# Transparent mode also mounts the proxy at /v1, so it's a drop-in replacement
# for the inference server at its own paths. Only meaningful when PROXY is on.
PROXY_TRANSPARENT = config.PROXY_TRANSPARENT
DB_PATH = config.DB_PATH
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
# Walking every process is the most expensive thing in a sample, so the host
# process table refreshes on its own multiple of the base interval.
HOST_PROC_EVERY = max(1, int(round(10.0 / INTERVAL)))

# How much history the in-memory ring holds, so a freshly opened tab has a
# populated live chart instead of a blank one that fills in over five minutes.
LIVE_WINDOW_S = 300

RANGES = {
    "5m": 300,
    "1h": 3600,
    "6h": 21600,
    "24h": 86400,
    "7d": 604800,
    "30d": 2592000,
}


class State:
    """Shared between the sampler thread and the request handlers."""

    def __init__(self):
        self.seq = 0
        self.latest: dict | None = None
        self.ring = collections.deque(maxlen=max(16, int(LIVE_WINDOW_S / INTERVAL)))
        self.info: dict = {}
        self.started = time.time()
        self.errors = 0
        self.samples = 0
        self.services: dict | None = None
        self.services_ts = 0.0
        self.host_procs: list = []


STATE = State()
STORE: Store | None = None
PROXY: ClassifyingProxy | None = None
FEED = PromptFeed()


def sampler(stop: threading.Event):
    """Poll metrics on a fixed cadence and persist each sample."""
    host = collector.HostCollector()
    last_maint = 0.0
    tick = 0

    while not stop.is_set():
        t0 = time.time()
        try:
            tick += 1
            want_procs = (tick % HOST_PROC_EVERY) == 1
            sample = collector.collect(host, want_host_procs=want_procs)
            # Carry the last table forward on the ticks that skip it, so the
            # panel holds its contents instead of blinking empty between
            # refreshes.
            if want_procs:
                STATE.host_procs = sample.get("host_procs") or []
            else:
                sample["host_procs"] = STATE.host_procs

            # Compact form for the live ring and the SSE stream: the chart
            # only needs scalars, not the full per-core and per-process lists.
            gpu = (sample["gpus"] or [{}])[0]
            h = sample["host"]
            STATE.ring.append({
                "ts": sample["ts"],
                "gpu_util": gpu.get("util"),
                "gpu_temp": gpu.get("temp"),
                "gpu_power": gpu.get("power"),
                "gpu_mem_used": gpu.get("mem_used"),
                "gpu_sm_clock": gpu.get("sm_clock"),
                "gpu_mem_clock": gpu.get("mem_clock"),
                "cpu": h.get("cpu"),
                "cpu_temp": h.get("cpu_temp"),
                "mem_used": h.get("mem_used"),
                "net_rx": h.get("net_rx"),
                "net_tx": h.get("net_tx"),
                "disk_read": h.get("disk_read"),
                "disk_write": h.get("disk_write"),
            })

            STATE.latest = sample
            STATE.seq += 1
            STATE.samples += 1

            if STORE:
                STORE.insert(sample)
        except Exception:
            STATE.errors += 1
            log.exception("sample failed")

        # Rollup and prune once a minute, on the sampler thread so there is
        # only ever one writer touching the database.
        if STORE and time.time() - last_maint > 60:
            last_maint = time.time()
            try:
                STORE.rollup()
                STORE.prune()
            except Exception:
                log.exception("maintenance failed")

        stop.wait(max(0.05, INTERVAL - (time.time() - t0)))


def services_sampler(stop: threading.Event):
    """Container and port inventory, on its own slower loop."""
    svc = services.ServiceCollector()
    while not stop.is_set():
        t0 = time.time()
        try:
            snapshot = svc.collect()
            STATE.services = snapshot
            STATE.services_ts = time.time()
            if STORE:
                STORE.insert_containers(
                    STATE.services_ts,
                    (snapshot.get("docker") or {}).get("containers") or [],
                )
        except Exception:
            log.exception("service inventory failed")
        stop.wait(max(1.0, SERVICES_INTERVAL - (time.time() - t0)))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global STORE
    STORE = Store(DB_PATH)
    try:
        rows = await asyncio.to_thread(STORE.backfill)
        log.info("rollup tier ready (%s minute rows)", rows)
    except Exception:
        log.exception("backfill failed -- long-range views may have gaps")

    STATE.info = collector.system_info()
    STATE.info["interval"] = INTERVAL
    STATE.info["ranges"] = list(RANGES.keys())

    caps = STATE.info.get("capabilities") or {}
    if caps.get("wsl"):
        log.info("WSL2 host detected -- per-process GPU accounting and CPU "
                 "temperature are not available from this platform")
    if config.HOST_PROC_ACTIVE:
        log.info("reading host metrics through %s", config.HOST_PROC)
    else:
        log.info("reading metrics from this namespace (no host /proc mounted)")

    if not STATE.info.get("gpu_present"):
        log.warning("nvidia-smi returned no GPUs -- serving host metrics only")
    else:
        log.info("watching %s x%s (unified memory: %s)",
                 STATE.info.get("gpu_name"), STATE.info.get("gpu_count"),
                 STATE.info.get("unified_memory"))

    global PROXY
    if PROXY_ENABLED:
        cfg = ProxyConfig()
        PROXY = ClassifyingProxy(cfg, FEED)
        log.info("vLLM proxy on -> upstream %s (classify=%s, sample=%.2f, transparent=%s)",
                 cfg.upstream, cfg.enabled, cfg.sample_rate, PROXY_TRANSPARENT)
        if cfg.classify_upstream != cfg.upstream:
            log.info("classifier -> separate upstream %s (model=%s)",
                     cfg.classify_upstream, cfg.classify_model or "<request's model>")
            if not cfg.classify_model:
                log.warning("classify upstream differs but SPARKBOARD_CLASSIFY_MODEL "
                            "is unset -- the classifier will be asked for the "
                            "request's model name, which it may not serve. Set "
                            "SPARKBOARD_CLASSIFY_MODEL to the small model's served name.")
        if PROXY_TRANSPARENT:
            log.info("transparent mode: /v1 is a drop-in for vLLM at this port")

    stop = threading.Event()
    threads = [
        threading.Thread(target=sampler, args=(stop,), daemon=True,
                         name="sparkboard-sampler"),
        threading.Thread(target=services_sampler, args=(stop,), daemon=True,
                         name="sparkboard-services"),
    ]
    for t in threads:
        t.start()
    log.info("sampling every %.1fs into %s", INTERVAL, DB_PATH)
    log.info("service inventory every %.0fs", SERVICES_INTERVAL)
    try:
        yield
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)
        if PROXY:
            await PROXY.aclose()
        if STORE:
            STORE.close()


app = FastAPI(title="Sparkboard", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def no_store(request, call_next):
    """
    Mark every API response uncacheable.

    Without this the polled endpoints carry no cache directives at all, and
    Safari in particular will serve a cached body to a repeated fetch of the
    same URL -- so the live charts keep moving over SSE while the polled
    tables sit frozen on their first response. Static assets are left alone.
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/") or path == "/metrics":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ------------------------------------------------------------------ routes

@app.get("/api/info")
async def api_info():
    return JSONResponse(STATE.info)


@app.get("/api/now")
async def api_now():
    if STATE.latest is None:
        return JSONResponse({"error": "no sample yet"}, status_code=503)
    return JSONResponse(STATE.latest)


@app.get("/api/live")
async def api_live():
    return JSONResponse({"interval": INTERVAL, "samples": list(STATE.ring)})


@app.get("/api/history")
async def api_history(
    range: str = Query("1h"),
    gpu: int = Query(0, ge=0),
):
    if range not in RANGES:
        return JSONResponse(
            {"error": f"unknown range {range!r}", "valid": list(RANGES)},
            status_code=400,
        )
    if STORE is None:
        return JSONResponse({"error": "store not ready"}, status_code=503)
    data = await asyncio.to_thread(STORE.history, RANGES[range], INTERVAL, gpu)
    data["label"] = range
    return JSONResponse(data)


@app.get("/api/services")
async def api_services():
    if STATE.services is None:
        return JSONResponse({"error": "inventory not ready"}, status_code=503)
    return JSONResponse({**STATE.services, "ts": STATE.services_ts})


@app.get("/api/containers/history")
async def api_container_history(range: str = Query("1h")):
    if range not in RANGES:
        return JSONResponse(
            {"error": f"unknown range {range!r}", "valid": list(RANGES)},
            status_code=400,
        )
    if STORE is None:
        return JSONResponse({"error": "store not ready"}, status_code=503)
    # Bucketed against the services cadence, not the telemetry one -- asking
    # for 2s buckets from a 10s sampler would just produce gaps.
    data = await asyncio.to_thread(
        STORE.container_history, RANGES[range], SERVICES_INTERVAL)
    data["label"] = range
    return JSONResponse(data)


@app.get("/api/prompts/feed")
async def api_prompt_feed(after: int = Query(0, ge=0)):
    """
    Recent activity labels and the category rollup. Contains no prompt text --
    see app/proxy.py for why that is true by construction, not by filtering.
    """
    return JSONResponse({
        "enabled": PROXY_ENABLED,
        "recent": FEED.recent(after=after, limit=100),
        "summary": FEED.summary(),
    })


# The proxy endpoints are mounted only when enabled.
#
# Two mounts, both forwarding to the same place:
#   /vllm/<path>  -- explicit, always present. Good for testing and for when
#                    the dashboard and proxy share a port but you want the
#                    proxy path to be unambiguous.
#   /v1/<path>    -- transparent drop-in. Present only in transparent mode
#                    (SPARKBOARD_PROXY_TRANSPARENT=1). This makes the proxy
#                    answer at the SAME paths vLLM does, so a client pointed at
#                    http://host:PORT/v1 is intercepted with no config change
#                    at all -- which is what you want when you can't touch
#                    every client (e.g. a teammate's Cline on another machine).
#
# Transparent mode is intended for when the proxy has taken over vLLM's own
# port and vLLM has moved to a private one. In that setup /v1 must belong to
# the proxy for existing clients to keep working.
if PROXY_ENABLED:
    @app.api_route("/vllm/{path:path}",
                   methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def vllm_proxy(path: str, request: Request):
        if PROXY is None:
            return JSONResponse({"error": "proxy not ready"}, status_code=503)
        return await PROXY.forward(request, path)

    if PROXY_TRANSPARENT:
        @app.api_route("/v1/{path:path}",
                       methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
        async def vllm_proxy_transparent(path: str, request: Request):
            if PROXY is None:
                return JSONResponse({"error": "proxy not ready"}, status_code=503)
            # Forward with the /v1 prefix restored, since that's what the
            # OpenAI API paths carry and what vLLM expects upstream.
            return await PROXY.forward(request, f"v1/{path}")


@app.get("/api/health")
async def api_health():
    return JSONResponse({
        "ok": STATE.latest is not None,
        "uptime": time.time() - STATE.started,
        "samples": STATE.samples,
        "errors": STATE.errors,
        "interval": INTERVAL,
        "store": STORE.stats() if STORE else None,
        "proxy": (PROXY.stats if PROXY else None),
        "proxy_enabled": PROXY_ENABLED,
        "proxy_transparent": PROXY_TRANSPARENT,
        "nvidia_smi": collector._NVIDIA_SMI or "not found",
        "gpus": STATE.info.get("gpu_count") or 0,
        "host_proc": config.HOST_PROC if config.HOST_PROC_ACTIVE else None,
        "docker_sock": config.DOCKER_SOCK,
        "capabilities": STATE.info.get("capabilities"),
        "db": DB_PATH,
    })


@app.get("/api/stream")
async def api_stream():
    """
    Server-sent events. One message per new sample.

    SSE rather than WebSocket because it survives an nginx proxy with a single
    `proxy_buffering off;` and reconnects on its own if the link drops -- no
    Upgrade-header dance, no client-side retry logic.
    """

    async def gen():
        last = -1
        keepalive = time.time()
        while True:
            if STATE.seq != last and STATE.latest is not None:
                last = STATE.seq
                payload = {
                    "seq": STATE.seq,
                    "sample": STATE.latest,
                    "point": STATE.ring[-1] if STATE.ring else None,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                keepalive = time.time()
            elif time.time() - keepalive > 15:
                # Comment frame keeps idle proxies from closing the connection.
                yield ": keepalive\n\n"
                keepalive = time.time()
            await asyncio.sleep(0.25)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # tells nginx not to buffer, belt and braces
            "Connection": "keep-alive",
        },
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus text exposition.

    Included so that adding Prometheus and Grafana later is a scrape-config
    entry rather than a rewrite. Nothing in the dashboard depends on it.
    """
    s = STATE.latest
    if s is None:
        return PlainTextResponse("# no sample yet\n", status_code=503)

    lines: list[str] = []

    def emit(name, help_text, value, labels=""):
        if value is None:
            return
        lines.append(f"# HELP sparkboard_{name} {help_text}")
        lines.append(f"# TYPE sparkboard_{name} gauge")
        lines.append(f"sparkboard_{name}{labels} {value}")

    for g in s.get("gpus") or []:
        lbl = f'{{gpu="{g.get("index", 0)}",name="{g.get("name", "GPU")}"}}'
        emit("gpu_utilization_percent", "GPU utilization", g.get("util"), lbl)
        emit("gpu_memory_used_bytes", "GPU memory in use",
             (g["mem_used"] * 1048576) if g.get("mem_used") is not None else None, lbl)
        emit("gpu_memory_total_bytes", "GPU memory total",
             (g["mem_total"] * 1048576) if g.get("mem_total") is not None else None, lbl)
        emit("gpu_temperature_celsius", "GPU temperature", g.get("temp"), lbl)
        emit("gpu_power_watts", "GPU power draw", g.get("power"), lbl)
        emit("gpu_sm_clock_hertz",
             "GPU SM clock",
             (g["sm_clock"] * 1_000_000) if g.get("sm_clock") is not None else None, lbl)

    h = s.get("host") or {}
    emit("cpu_utilization_percent", "CPU utilization", h.get("cpu"))
    emit("cpu_temperature_celsius", "CPU temperature", h.get("cpu_temp"))
    emit("memory_used_bytes", "Host memory in use", h.get("mem_used"))
    emit("memory_total_bytes", "Host memory total", h.get("mem_total"))
    emit("load1", "1-minute load average", h.get("load1"))
    emit("network_receive_bytes_per_second", "Network receive rate", h.get("net_rx"))
    emit("network_transmit_bytes_per_second", "Network transmit rate", h.get("net_tx"))
    emit("disk_read_bytes_per_second", "Disk read rate", h.get("disk_read"))
    emit("disk_write_bytes_per_second", "Disk write rate", h.get("disk_write"))
    emit("sample_errors_total", "Failed collection attempts", STATE.errors)

    return PlainTextResponse("\n".join(lines) + "\n")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# Mounted last so the API routes above take precedence.
app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn

    uvicorn.run(
        app,
        host=config.BIND,
        port=config.PORT,
        log_level=config.LOGLEVEL.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
