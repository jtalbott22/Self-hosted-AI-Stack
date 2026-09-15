"""
Configuration, resolved once at import.

Two things live here rather than being scattered through the modules.

First, environment names. Sparkboard grew up as a systemd service on bare
metal, where everything was prefixed SPARKBOARD_*. It now also runs as a
container, where the shorter unprefixed names (DB_PATH, SAMPLE_INTERVAL,
HOST_PROC) are conventional and were already in use. Both are accepted, the
prefixed one winning, so an existing compose file or unit keeps working
without edits.

Second, and the reason this module is imported before anything else touches
psutil: when Sparkboard runs inside a container it has to be pointed at the
host's /proc, or it reports on its own namespace -- its own two processes, its
own loopback-only socket table -- and calls that the machine. Setting
psutil.PROCFS_PATH has to happen before the first psutil call, so it happens
here, and app/__init__.py imports this module first to guarantee the ordering.
"""

from __future__ import annotations

import os


def _env(*names, default=None):
    """First name that is set and non-empty wins."""
    for n in names:
        v = os.environ.get(n)
        if v is not None and v.strip() != "":
            return v.strip()
    return default


def _num(*names, default=0.0, cast=float):
    raw = _env(*names)
    if raw is None:
        return cast(default)
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return cast(default)


def _flag(*names, default=False):
    raw = _env(*names)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- sampling

INTERVAL = max(0.5, _num("SPARKBOARD_INTERVAL", "SAMPLE_INTERVAL", default=2.0))
SERVICES_INTERVAL = max(
    2.0, _num("SPARKBOARD_SERVICES_INTERVAL", "SERVICES_INTERVAL", default=10.0))

# ---------------------------------------------------------------- storage

DB_PATH = _env("SPARKBOARD_DB", "DB_PATH", default="/var/lib/sparkboard/metrics.db")
# Raw samples are what the 5m..24h views read. A day and change is enough to
# cover "what happened overnight" without the table growing without bound.
RAW_RETENTION_HOURS = _num(
    "SPARKBOARD_RAW_RETENTION_HOURS", "RAW_RETENTION_HOURS", default=26.0)
# Minute rollups back the 7d and 30d views. The container build defaults this
# lower than the bare-metal one only if RETENTION_DAYS is set explicitly.
RETENTION_DAYS = _num("SPARKBOARD_RETENTION_DAYS", "RETENTION_DAYS", default=120.0)

# ----------------------------------------------------------------- server

BIND = _env("SPARKBOARD_BIND", "BIND", default="127.0.0.1")
PORT = int(_num("SPARKBOARD_PORT", "PORT", default=9101, cast=int))
LOGLEVEL = (_env("SPARKBOARD_LOGLEVEL", "LOGLEVEL", default="INFO") or "INFO").upper()

# ------------------------------------------------------------------- host

# Where the host's /proc is mounted, when running containerised. Empty or
# missing means "use my own", which is correct on bare metal.
HOST_PROC = _env("SPARKBOARD_HOST_PROC", "HOST_PROC", default="")
# Name shown in the header. A container's hostname is a random hex string, so
# without this the dashboard would identify the machine as "3f9c1a0b4e77".
HOST_LABEL = _env("SPARKBOARD_HOST_LABEL", "HOST_LABEL", default="")
# Filesystems to report. Comma-separated; a path may be given as
# "/host/disks/c=C:" to relabel it for display.
DISK_PATHS = [p.strip() for p in
              (_env("SPARKBOARD_DISK_PATHS", "DISK_PATHS", default="/") or "/").split(",")
              if p.strip()]

# -------------------------------------------------------------------- GPU

NVIDIA_SMI_PATH = _env("SPARKBOARD_NVIDIA_SMI", "NVIDIA_SMI_PATH", default="")
SMI_TIMEOUT = max(1.0, _num("SPARKBOARD_SMI_TIMEOUT", "SMI_TIMEOUT", default=8.0))

# ----------------------------------------------------------------- docker

# Unix socket for the Docker Engine API. Preferred over shelling out to the
# docker CLI, which a slim container image does not have.
DOCKER_SOCK = _env("SPARKBOARD_DOCKER_SOCK", "DOCKER_SOCK", default="/var/run/docker.sock")
DOCKER_ENABLED = _flag("SPARKBOARD_DOCKER", "DOCKER_ENABLED", default=True)

# ------------------------------------------------------------------ proxy

PROXY_ENABLED = _flag("SPARKBOARD_PROXY", default=False)
PROXY_TRANSPARENT = _flag("SPARKBOARD_PROXY_TRANSPARENT", default=False)


# --------------------------------------------------------------- side effect

def _point_psutil_at_host():
    """
    Redirect psutil at the host's procfs, if one was mounted.

    This governs more than the process list: psutil reads memory, boot time,
    network counters, and the socket table through the same path, so with it
    set the container reports the machine it runs on rather than itself.
    """
    if not HOST_PROC:
        return False
    if not os.path.isdir(HOST_PROC):
        return False
    try:
        import psutil
    except ImportError:
        return False
    psutil.PROCFS_PATH = HOST_PROC
    return True


HOST_PROC_ACTIVE = _point_psutil_at_host()
