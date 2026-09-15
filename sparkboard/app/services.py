"""
Container and listening-port inventory.

Answers two questions the GPU charts can't: what is running, and what is it
reachable on.

Sampled on a slower loop than the main metrics. `docker stats` samples CPU
over a short interval internally, so a call costs a second or two -- far too
slow for the 2s telemetry cadence, and pointless at that resolution anyway
since containers don't come and go every two seconds.

Two ways in to Docker, picked automatically:

  API   the Engine's REST interface over /var/run/docker.sock. Preferred, and
        the only one that works from inside a slim container, which has no
        docker binary to shell out to. It also returns raw counters instead
        of formatted strings, so nothing has to be parsed back out of "1.04GiB".
  CLI   `docker ps` / `docker stats`. The fallback for a bare-metal install
        where the socket isn't reachable but the CLI is.

Both produce the same row shape, so everything downstream is unaware of which
one answered.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config

log = logging.getLogger("sparkboard.services")

_DOCKER = shutil.which("docker")

# Ordered: first match wins, so more specific patterns come first.
_TAGS = [
    ("vllm", "vLLM"),
    ("llama[-_.]?cpp|llama[-_]server|\\bllama\\b", "llama.cpp"),
    ("comfy", "ComfyUI"),
    ("open[-_]webui", "Open WebUI"),
    ("librechat", "LibreChat"),
    ("jupyter", "Jupyter"),
    ("searxng", "SearXNG"),
    ("meilisearch", "MeiliSearch"),
    ("mongod", "MongoDB"),
    ("playwright", "Playwright"),
    ("ollama", "Ollama"),
    ("uvicorn", "uvicorn"),
    ("gunicorn", "gunicorn"),
    ("hypercorn", "hypercorn"),
    ("docker-proxy", "docker"),
    ("containerd", "containerd"),
    ("nginx", "nginx"),
    (r"\bnode\b", "node"),
    ("sshd", "ssh"),
    ("postgres", "postgres"),
    ("redis", "redis"),
    ("sparkboard", "sparkboard"),
]

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([KMGTP]?i?B)\s*$", re.I)
_UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "PB": 1e15,
          "KIB": 1024, "MIB": 1024 ** 2, "GIB": 1024 ** 3,
          "TIB": 1024 ** 4, "PIB": 1024 ** 5}


def _size(text):
    """
    Parse a docker size string to bytes.

    Docker mixes conventions in the same output: memory comes back binary
    (MiB, GiB) while network and block I/O come back decimal (kB, MB). The
    suffix decides which, so 'i' is not cosmetic here.
    """
    if not text:
        return None
    m = _SIZE_RE.match(text)
    if not m:
        return None
    try:
        return float(m.group(1)) * _UNITS[m.group(2).upper()]
    except (KeyError, ValueError):
        return None


def _pair(text):
    """Split docker's 'a / b' fields into two byte counts."""
    if not text or "/" not in text:
        return None, None
    a, b = text.split("/", 1)
    return _size(a), _size(b)


def _pct(text):
    if not text:
        return None
    try:
        return float(text.strip().rstrip("%"))
    except ValueError:
        return None


def _run(args, timeout=20):
    try:
        out = subprocess.run(args, capture_output=True, text=True,
                             timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return None, str(exc)
    if out.returncode != 0:
        return None, (out.stderr or "").strip()[:200] or f"exit {out.returncode}"
    return out.stdout, None


def _jsonl(text):
    rows = []
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def parse_ports(text):
    """
    Turn docker's port string into deduplicated host->container mappings.

    Docker lists the same publication once per address family, so
    '0.0.0.0:3000->8080/tcp, :::3000->8080/tcp' is one mapping shown twice.
    """
    if not text:
        return []
    seen = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^(?:(.*):)?(\d+)->(\d+)/(\w+)$", chunk)
        if m:
            host_port, cont_port, proto = int(m.group(2)), int(m.group(3)), m.group(4)
            seen.setdefault((host_port, cont_port, proto),
                            {"host": host_port, "container": cont_port, "proto": proto})
            continue
        m = re.match(r"^(\d+)/(\w+)$", chunk)   # exposed but not published
        if m:
            cont_port, proto = int(m.group(1)), m.group(2)
            seen.setdefault((None, cont_port, proto),
                            {"host": None, "container": cont_port, "proto": proto})
    return sorted(seen.values(), key=lambda p: (p["host"] is None, p["host"] or p["container"]))


# ------------------------------------------------------------- engine API

_api_lock = threading.Lock()
_api_client = None
_api_failed = False


def _blank_row(cid, name, image, state, status, ports):
    return {
        "id": (cid or "")[:12],
        "name": name or "",
        "image": image or "",
        "state": state or "",
        "status": status or "",
        "ports": ports,
        "cpu": None, "mem_used": None, "mem_limit": None, "mem_pct": None,
        "net_rx": None, "net_tx": None, "block_read": None, "block_write": None,
        "pids": None,
    }


def _docker_api():
    """
    Lazily built httpx client bound to the Docker socket.

    No API version is pinned in the path: the daemon then answers with its own
    newest supported version, which is what keeps this working across engine
    upgrades without a version table to maintain here.
    """
    global _api_client, _api_failed
    if _api_client is not None or _api_failed:
        return _api_client
    with _api_lock:
        if _api_client is not None or _api_failed:
            return _api_client
        sock = config.DOCKER_SOCK
        if not sock or not os.path.exists(sock):
            _api_failed = True
            return None
        try:
            import httpx
            _api_client = httpx.Client(
                transport=httpx.HTTPTransport(uds=sock),
                base_url="http://docker",
                timeout=httpx.Timeout(25.0, connect=5.0),
            )
        except Exception as exc:
            log.debug("docker api client unavailable: %s", exc)
            _api_failed = True
            return None
    return _api_client


def api_ports(entries):
    """
    Normalise the Engine's port array.

    Same publication appears once per address family -- 0.0.0.0 and :: -- so
    the pair is deduplicated the same way the CLI string form is.
    """
    seen = {}
    for e in entries or []:
        proto = (e.get("Type") or "tcp").lower()
        private = e.get("PrivatePort")
        public = e.get("PublicPort")
        if private is None:
            continue
        seen.setdefault((public, private, proto),
                        {"host": public, "container": private, "proto": proto})
    return sorted(seen.values(),
                  key=lambda p: (p["host"] is None, p["host"] or p["container"]))


def _api_cpu_percent(stats):
    """
    The same arithmetic `docker stats` does, on the same raw counters.

    Both deltas are needed, which is why the snapshot is fetched with
    stream=false rather than one-shot: one-shot returns an empty precpu block
    and every container would read 0%.
    """
    try:
        cpu = stats["cpu_stats"]
        pre = stats.get("precpu_stats") or {}
        cpu_delta = cpu["cpu_usage"]["total_usage"] - \
            (pre.get("cpu_usage") or {}).get("total_usage", 0)
        sys_delta = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
        if cpu_delta <= 0 or sys_delta <= 0:
            return None
        ncpu = cpu.get("online_cpus") or len(
            (cpu.get("cpu_usage") or {}).get("percpu_usage") or []) or 1
        return round((cpu_delta / sys_delta) * ncpu * 100.0, 2)
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def _api_memory(stats):
    """
    Memory in use and the limit, matching what `docker stats` prints.

    Page cache is subtracted: the kernel counts it against the cgroup but it
    is reclaimable, and leaving it in makes an idle container that once read a
    large file look permanently fat. The field is named inactive_file under
    cgroup v2 and total_inactive_file under v1.
    """
    mem = stats.get("memory_stats") or {}
    usage = mem.get("usage")
    if usage is None:
        return None, None, None
    detail = mem.get("stats") or {}
    cache = detail.get("inactive_file")
    if cache is None:
        cache = detail.get("total_inactive_file", 0)
    used = max(0, usage - (cache or 0))
    limit = mem.get("limit") or None
    pct = round(used / limit * 100, 2) if limit else None
    return float(used), (float(limit) if limit else None), pct


def _api_io(stats):
    net_rx = net_tx = None
    nets = stats.get("networks")
    if isinstance(nets, dict) and nets:
        net_rx = float(sum(n.get("rx_bytes", 0) for n in nets.values()))
        net_tx = float(sum(n.get("tx_bytes", 0) for n in nets.values()))

    read = write = None
    entries = (stats.get("blkio_stats") or {}).get("io_service_bytes_recursive")
    if entries:
        read = float(sum(e.get("value", 0) for e in entries
                         if (e.get("op") or "").lower() == "read"))
        write = float(sum(e.get("value", 0) for e in entries
                          if (e.get("op") or "").lower() == "write"))
    return net_rx, net_tx, read, write


def collect_docker_api():
    """Container inventory via the Engine API. None means "not usable here"."""
    client = _docker_api()
    if client is None:
        return None

    try:
        resp = client.get("/containers/json")
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:
        # Distinguish "socket not mounted" from "mounted but not permitted":
        # they need different fixes and look the same from the dashboard.
        text = str(exc).lower()
        if "permission denied" in text:
            hint = ("permission denied on the docker socket -- the container "
                    "user cannot read " + config.DOCKER_SOCK)
        elif "connect" in text or "refused" in text:
            hint = f"cannot reach the docker socket at {config.DOCKER_SOCK}"
        else:
            hint = f"docker api error: {exc}"[:200]
        return {"available": False, "error": hint, "containers": []}

    containers = []
    for row in rows:
        names = row.get("Names") or []
        name = (names[0] if names else "").lstrip("/")
        containers.append(_blank_row(
            row.get("Id"), name, row.get("Image"), row.get("State"),
            row.get("Status"), api_ports(row.get("Ports")),
        ))
    if not containers:
        return {"available": True, "error": None, "containers": []}

    # Each stats call blocks for about a second while the daemon takes its
    # second CPU reading, so serially this would cost a second per container
    # and overrun the sampling interval on a busy host. In parallel the whole
    # sweep costs about as much as the slowest single call.
    ids = [(c, row.get("Id")) for c, row in zip(containers, rows)]

    def one(pair):
        container, cid = pair
        try:
            r = client.get(f"/containers/{cid}/stats",
                           params={"stream": "false"})
            r.raise_for_status()
            return container, r.json()
        except Exception as exc:
            log.debug("stats for %s failed: %s", container["name"], exc)
            return container, None

    stats_error = None
    try:
        with ThreadPoolExecutor(max_workers=min(8, len(ids))) as pool:
            results = list(pool.map(one, ids))
    except Exception as exc:
        results = []
        stats_error = str(exc)[:200]

    for container, stats in results:
        if not stats:
            continue
        used, limit, pct = _api_memory(stats)
        rx, tx, read, write = _api_io(stats)
        container.update({
            "cpu": _api_cpu_percent(stats),
            "mem_used": used, "mem_limit": limit, "mem_pct": pct,
            "net_rx": rx, "net_tx": tx,
            "block_read": read, "block_write": write,
            "pids": (stats.get("pids_stats") or {}).get("current"),
        })

    containers.sort(key=lambda c: c.get("mem_used") or 0, reverse=True)
    out = {"available": True, "error": None, "containers": containers}
    if stats_error:
        out["stats_error"] = stats_error
    return out


def collect_docker_cli():
    """Running containers with their current resource usage, via the CLI."""
    if not _DOCKER:
        return {"available": False, "error": "docker not installed", "containers": []}

    ps_out, err = _run([_DOCKER, "ps", "--format", "{{json .}}"], timeout=15)
    if ps_out is None:
        # Almost always a permissions problem on /var/run/docker.sock.
        hint = err or "docker ps failed"
        if "permission denied" in hint.lower():
            hint = "permission denied on the docker socket"
        return {"available": False, "error": hint, "containers": []}

    containers = []
    for row in _jsonl(ps_out):
        containers.append({
            "id": (row.get("ID") or "")[:12],
            "name": row.get("Names") or "",
            "image": row.get("Image") or "",
            "state": row.get("State") or "",
            "status": row.get("Status") or "",
            "ports": parse_ports(row.get("Ports")),
            "cpu": None, "mem_used": None, "mem_limit": None, "mem_pct": None,
            "net_rx": None, "net_tx": None, "block_read": None, "block_write": None,
            "pids": None,
        })

    if not containers:
        return {"available": True, "error": None, "containers": []}

    # Stats are best-effort: if the daemon is slow the inventory is still worth
    # showing, just without the usage columns.
    stats_out, serr = _run([_DOCKER, "stats", "--no-stream", "--format", "{{json .}}"],
                           timeout=25)
    if stats_out is None:
        log.debug("docker stats unavailable: %s", serr)
        return {"available": True, "error": None, "containers": containers,
                "stats_error": serr or "docker stats timed out"}

    by_name = {c["name"]: c for c in containers}
    by_id = {c["id"]: c for c in containers}
    for row in _jsonl(stats_out):
        c = by_name.get(row.get("Name")) or by_id.get((row.get("ID") or "")[:12])
        if not c:
            continue
        mem_used, mem_limit = _pair(row.get("MemUsage"))
        net_rx, net_tx = _pair(row.get("NetIO"))
        blk_r, blk_w = _pair(row.get("BlockIO"))
        c.update({
            "cpu": _pct(row.get("CPUPerc")),
            "mem_used": mem_used,
            "mem_limit": mem_limit,
            "mem_pct": _pct(row.get("MemPerc")),
            "net_rx": net_rx, "net_tx": net_tx,
            "block_read": blk_r, "block_write": blk_w,
            "pids": int(row["PIDs"]) if str(row.get("PIDs", "")).isdigit() else None,
        })

    containers.sort(key=lambda c: c.get("mem_used") or 0, reverse=True)
    return {"available": True, "error": None, "containers": containers}


def collect_docker():
    """
    Container inventory, API first and CLI second.

    The API is tried on every pass rather than latched at startup, so a daemon
    that was down when Sparkboard booted is picked up when it comes back.
    Only a genuinely unusable socket falls through to the CLI.
    """
    if not config.DOCKER_ENABLED:
        return {"available": False, "error": "docker inventory disabled",
                "containers": []}

    result = collect_docker_api()
    if result is not None and (result.get("available") or not _DOCKER):
        return result
    cli = collect_docker_cli()
    # Prefer whichever actually answered; if both failed, report the API's
    # reason, since that is the path this build is normally installed on.
    if cli.get("available") or result is None:
        return cli
    return result


def _tag_for(cmdline, name):
    haystack = f"{cmdline} {name}".lower()
    for pattern, label in _TAGS:
        if re.search(pattern, haystack):
            return label
    return None


def collect_listeners(container_ports=None):
    """
    TCP sockets in LISTEN state, attributed to the process holding them.

    Attribution needs to read /proc/<pid>/fd for processes owned by other
    users. Unprivileged, that fails silently: the socket is still listed but
    pid comes back None. The service unit grants CAP_SYS_PTRACE and
    CAP_DAC_READ_SEARCH for this, and the container equivalent is cap_add:
    SYS_PTRACE; when they're missing the ports are still shown, flagged as
    unattributed, rather than quietly omitted.

    Which socket table gets read is decided by psutil's procfs path, set once
    in config. Containerised without a host /proc mounted, that table is the
    container's own -- one port, its own -- which is a true answer to a
    question nobody asked. The host_scope flag says which it was.
    """
    import psutil

    container_ports = container_ports or {}
    try:
        conns = psutil.net_connections(kind="inet")
    except psutil.AccessDenied:
        return {"privileged": False, "error": "cannot enumerate sockets",
                "ports": [], "host_scope": config.HOST_PROC_ACTIVE}

    listening = [c for c in conns
                 if c.status == psutil.CONN_LISTEN and c.laddr]

    # One service bound to both 0.0.0.0 and :: appears twice for one port.
    merged = {}
    for c in listening:
        key = (c.laddr.port, c.pid)
        entry = merged.setdefault(key, {
            "port": c.laddr.port, "pid": c.pid, "addrs": set(),
        })
        entry["addrs"].add(c.laddr.ip)

    proc_cache = {}
    resolved = 0
    out = []
    for entry in merged.values():
        pid = entry["pid"]
        name = cmdline = user = None
        if pid:
            resolved += 1
            if pid not in proc_cache:
                try:
                    p = psutil.Process(pid)
                    with p.oneshot():
                        # Arguments can contain newlines and runs of spaces
                        # (anything launched with `python -c`, for one), which
                        # would wreck a table cell. Collapse to single spaces.
                        raw = " ".join(p.cmdline())
                        flat = re.sub(r"\s+", " ", raw).strip()
                        proc_cache[pid] = (
                            p.name(),
                            flat[:220] or p.name(),
                            p.username(),
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    proc_cache[pid] = (None, None, None)
            name, cmdline, user = proc_cache[pid]

        addrs = sorted(entry["addrs"])
        scope = "all interfaces"
        if addrs and all(a in ("127.0.0.1", "::1") for a in addrs):
            scope = "localhost"

        tag = _tag_for(cmdline or "", name or "")
        container = container_ports.get(entry["port"])
        if container:
            tag = tag or "docker"

        out.append({
            "port": entry["port"],
            "pid": pid,
            "process": name,
            "cmdline": cmdline,
            "user": user,
            "scope": scope,
            "addrs": addrs,
            "tag": tag,
            "container": container,
        })

    out.sort(key=lambda r: r["port"])
    return {
        "privileged": resolved > 0 or not out,
        "error": None,
        "ports": out,
        "unattributed": sum(1 for r in out if not r["pid"]),
        "host_scope": config.HOST_PROC_ACTIVE,
    }


class ServiceCollector:
    """
    Stateful wrapper around the inventory.

    Docker reports NetIO and BlockIO as totals accumulated since the container
    started, so plotting them directly draws a line that only ever goes up.
    Holding the previous reading turns them into per-second rates, which is
    what you actually want on a chart. The table still shows the cumulative
    figure, because that's the number `docker stats` shows and the one people
    expect to see there.
    """

    def __init__(self):
        self._prev = {}

    def _rates(self, c, now):
        key = c["name"]
        prev = self._prev.get(key)
        cur = (now, c.get("net_rx"), c.get("net_tx"),
               c.get("block_read"), c.get("block_write"))
        self._prev[key] = cur

        for field in ("net_rx_rate", "net_tx_rate", "block_read_rate", "block_write_rate"):
            c[field] = None
        if not prev:
            return
        dt = now - prev[0]
        if dt <= 0:
            return
        for i, field in enumerate(
                ("net_rx_rate", "net_tx_rate", "block_read_rate", "block_write_rate"),
                start=1):
            a, b = prev[i], cur[i]
            # A container restart resets its counters; a negative delta means
            # that happened, and reporting it as a huge negative rate would be
            # worse than reporting nothing.
            if a is None or b is None or b < a:
                continue
            c[field] = (b - a) / dt

    def collect(self):
        docker = collect_docker()
        now = time.time()
        live = set()
        for c in docker.get("containers") or []:
            live.add(c["name"])
            self._rates(c, now)
        # Drop state for containers that are gone, so a long-lived process
        # doesn't accumulate an entry per container ever seen.
        for stale in set(self._prev) - live:
            self._prev.pop(stale, None)

        port_owner = {}
        for c in docker.get("containers") or []:
            for p in c.get("ports") or []:
                if p.get("host"):
                    port_owner[p["host"]] = c["name"]
        return {"docker": docker, "listeners": collect_listeners(port_owner)}


def collect():
    """Stateless one-shot, for callers that don't need rates."""
    return ServiceCollector().collect()
