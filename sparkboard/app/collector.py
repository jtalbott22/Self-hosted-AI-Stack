"""
Metrics collection for Sparkboard.

Pulls GPU metrics from `nvidia-smi` and host metrics from psutil.

Design notes
------------
Grace-Blackwell parts (GB10 / DGX Spark) legitimately return "[N/A]" or
"[Not Supported]" for several nvidia-smi fields -- fan speed and power limit
are the usual suspects, since the module is passively managed and doesn't
expose a discrete board power cap. Every field is therefore parsed through
`_num()`, which returns None rather than raising. The frontend renders None
as an em-dash instead of a zero, so a missing sensor never masquerades as a
real reading of 0.

If nvidia-smi is absent entirely the collector still runs and returns host
metrics only, so the service comes up on a machine without a GPU.

WSL2
----
Under WSL2 the driver is reached through a shim that Windows exposes to the
VM. Two things differ there and both are reported rather than papered over:
`--query-compute-apps` returns nothing, because the WSL driver does not do
per-process accounting, and there are no hwmon sensors, so host CPU
temperature is unavailable. `capabilities()` states which of these are
missing so the UI can say "not reported here" instead of showing an empty
table that looks like an idle machine.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import time

from . import config

log = logging.getLogger("sparkboard.collector")

# Values nvidia-smi uses to mean "no reading available".
_NA = {"", "n/a", "[n/a]", "[not supported]", "not supported",
       "[unknown error]", "unknown", "[insufficient permissions]"}

# Order matters: it maps positionally onto the CSV that nvidia-smi returns.
GPU_FIELDS = [
    "index",
    "name",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.current.graphics",
    "fan.speed",
    "pstate",
]

def _find_smi():
    """
    Locate nvidia-smi.

    PATH is the normal answer, and the one the NVIDIA container toolkit
    arranges by injecting the binary at /usr/bin. The WSL path is the fallback
    for when the toolkit is absent: Windows exposes the driver to the VM under
    /usr/lib/wsl/lib, which can be bind-mounted into a container with no
    toolkit involved at all. An explicit NVIDIA_SMI_PATH overrides both.
    """
    for cand in (config.NVIDIA_SMI_PATH, "nvidia-smi", "/usr/bin/nvidia-smi",
                 "/usr/local/bin/nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
        if not cand:
            continue
        if os.sep in cand:
            if os.path.exists(cand) and os.access(cand, os.X_OK):
                return cand
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


_NVIDIA_SMI = _find_smi()
_BOOT_TIME = None


def is_wsl() -> bool:
    """
    True when the kernel is the one Microsoft ships for WSL.

    Checked against the host's procfs when one is mounted, so a container on a
    WSL host answers for the host and not for itself.
    """
    try:
        path = f"{config.HOST_PROC}/sys/kernel/osrelease" if config.HOST_PROC_ACTIVE \
            else "/proc/sys/kernel/osrelease"
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            release = fh.read()
    except OSError:
        release = platform.release()
    return "microsoft" in (release or "").lower()


# Parts that draw from system RAM rather than a board-local framebuffer. Used
# only to confirm an inference, never to make one on its own.
_UNIFIED_HINTS = ("gb10", "grace", "orin", "thor", "jetson", "igpu", "integrated")


def _num(raw):
    """Parse a numeric nvidia-smi cell, returning None for absent readings."""
    if raw is None:
        return None
    s = raw.strip()
    if s.lower() in _NA:
        return None
    # Strip any trailing unit nvidia-smi left behind despite `nounits`.
    m = re.match(r"^-?\d+(\.\d+)?", s)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    return v


def _text(raw):
    if raw is None:
        return None
    s = raw.strip()
    return None if s.lower() in _NA else s


def _run(args, timeout=None):
    try:
        out = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else config.SMI_TIMEOUT,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("command failed: %s (%s)", " ".join(args), exc)
        return None
    if out.returncode != 0:
        log.warning("command exited %s: %s", out.returncode, out.stderr.strip()[:200])
        return None
    return out.stdout


def nvidia_available() -> bool:
    return _NVIDIA_SMI is not None


def driver_info() -> dict:
    """Driver and CUDA version. Queried once at startup, not per sample."""
    info = {"driver_version": None, "cuda_version": None}
    if not _NVIDIA_SMI:
        return info
    out = _run([
        _NVIDIA_SMI,
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ])
    if out:
        first = out.strip().splitlines()
        if first:
            info["driver_version"] = _text(first[0])
    # CUDA version only appears in the human-readable header table.
    out = _run([_NVIDIA_SMI])
    if out:
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            info["cuda_version"] = m.group(1)
    return info


def collect_gpus() -> list:
    """One row per GPU. Empty list if nvidia-smi is unavailable or failed."""
    if not _NVIDIA_SMI:
        return []
    out = _run([
        _NVIDIA_SMI,
        "--query-gpu=" + ",".join(GPU_FIELDS),
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []

    gpus = []
    for line in out.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < len(GPU_FIELDS):
            cells += [""] * (len(GPU_FIELDS) - len(cells))
        c = dict(zip(GPU_FIELDS, cells))

        idx = _num(c["index"])
        mem_total = _num(c["memory.total"])
        mem_used = _num(c["memory.used"])
        # Prefer the SM clock; fall back to the graphics clock when the part
        # reports one but not the other.
        sm_clock = _num(c["clocks.current.sm"])
        if sm_clock is None:
            sm_clock = _num(c["clocks.current.graphics"])

        gpus.append({
            "index": int(idx) if idx is not None else 0,
            "name": _text(c["name"]) or "GPU",
            "uuid": _text(c["uuid"]),
            "util": _num(c["utilization.gpu"]),
            "mem_util": _num(c["utilization.memory"]),
            "mem_total": mem_total,          # MiB
            "mem_used": mem_used,            # MiB
            "mem_free": _num(c["memory.free"]),
            "temp": _num(c["temperature.gpu"]),
            "power": _num(c["power.draw"]),
            "power_limit": _num(c["power.limit"]),
            "sm_clock": sm_clock,            # MHz
            "mem_clock": _num(c["clocks.current.memory"]),
            "fan": _num(c["fan.speed"]),
            "pstate": _text(c["pstate"]),
        })
    return gpus


def collect_processes(limit=20) -> list:
    """Processes currently holding GPU memory."""
    if not _NVIDIA_SMI:
        return []
    out = _run([
        _NVIDIA_SMI,
        "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []

    procs = []
    for line in out.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 4:
            continue
        pid = _num(cells[0])
        if pid is None:
            continue
        name = _text(cells[1]) or "?"
        procs.append({
            "pid": int(pid),
            "name": os.path.basename(name),
            "cmd": name,
            "gpu_uuid": _text(cells[2]),
            "mem": _num(cells[3]),  # MiB
        })
    procs.sort(key=lambda p: p["mem"] or 0, reverse=True)
    return procs[:limit]


def reconcile_memory(gpus: list, procs: list, host_mem_total: int):
    """
    Fill in GPU memory on parts that don't report it.

    GB10 has no discrete framebuffer, so `nvidia-smi --query-gpu=memory.total`
    and `memory.used` come back as [N/A] -- there is no separate pool for them
    to describe. The per-process query still works, so allocation is
    reconstructed by summing it, and the pool size is the host's RAM.

    A missing memory.total is itself the tell: a discrete card always reports
    its framebuffer size. Both substitutions are flagged with *_inferred so
    the UI can say where the number came from rather than implying nvidia-smi
    handed it over.
    """
    if not gpus:
        return

    for g in gpus:
        if g.get("mem_total") is None and host_mem_total:
            g["mem_total"] = host_mem_total / (1024 * 1024)  # MiB
            g["mem_total_inferred"] = True

        if g.get("mem_used") is None and procs:
            # Match on UUID when nvidia-smi gives us one; fall back to summing
            # everything only when there's a single GPU and no ambiguity.
            uuid = g.get("uuid")
            if uuid and any(p.get("gpu_uuid") for p in procs):
                mine = [p for p in procs if p.get("gpu_uuid") == uuid]
            elif len(gpus) == 1:
                mine = procs
            else:
                mine = []
            if mine:
                g["mem_used"] = sum(p["mem"] for p in mine if p.get("mem") is not None)
                g["mem_used_inferred"] = True

        if (g.get("mem_free") is None
                and g.get("mem_total") is not None
                and g.get("mem_used") is not None):
            g["mem_free"] = max(0.0, g["mem_total"] - g["mem_used"])


class HostCollector:
    """
    Host metrics. Holds the previous counter reading so network and disk
    throughput come out as per-second rates rather than monotonic totals.
    """

    def __init__(self):
        import psutil

        self.psutil = psutil
        self._last = None
        # Prime the CPU percent counter so the first real sample isn't 0.
        psutil.cpu_percent(interval=None, percpu=True)

    def collect(self) -> dict:
        ps = self.psutil
        now = time.time()

        vm = ps.virtual_memory()
        sw = ps.swap_memory()
        per_core = ps.cpu_percent(interval=None, percpu=True)
        cpu_pct = sum(per_core) / len(per_core) if per_core else 0.0

        try:
            load1, load5, load15 = os.getloadavg()
        except (OSError, AttributeError):
            load1 = load5 = load15 = None

        net_rx = net_tx = disk_r = disk_w = None
        try:
            net = ps.net_io_counters()
            dio = ps.disk_io_counters()
            cur = {
                "t": now,
                "rx": net.bytes_recv if net else None,
                "tx": net.bytes_sent if net else None,
                "dr": dio.read_bytes if dio else None,
                "dw": dio.write_bytes if dio else None,
            }
            if self._last:
                dt = cur["t"] - self._last["t"]
                if dt > 0:
                    def rate(key):
                        a, b = self._last.get(key), cur.get(key)
                        if a is None or b is None or b < a:
                            return None
                        return (b - a) / dt
                    net_rx, net_tx = rate("rx"), rate("tx")
                    disk_r, disk_w = rate("dr"), rate("dw")
            self._last = cur
        except Exception as exc:  # counters are optional, never fatal
            log.debug("io counters unavailable: %s", exc)

        cpu_temp = self._cpu_temp()

        disks = collect_disks()
        # The first configured filesystem is the one that goes to the
        # time-series store; the rest are current-state only.
        disk_used = disks[0]["used"] if disks else None
        disk_total = disks[0]["total"] if disks else None

        freq = None
        try:
            f = ps.cpu_freq()
            freq = round(f.current) if f and f.current else None
        except (OSError, AttributeError, NotImplementedError):
            freq = None

        return {
            "cpu": cpu_pct,
            "cpu_cores": per_core,
            "cpu_temp": cpu_temp,
            "mem_used": vm.total - vm.available,  # bytes
            "mem_total": vm.total,
            "swap_used": sw.used,
            "swap_total": sw.total,
            "load1": load1,
            "load5": load5,
            "load15": load15,
            "net_rx": net_rx,
            "net_tx": net_tx,
            "disk_read": disk_r,
            "disk_write": disk_w,
            "disk_used": disk_used,
            "disk_total": disk_total,
            "disks": disks,
            "cpu_freq": freq,
        }

    def _cpu_temp(self):
        """
        Best-effort CPU temperature. On ARM hosts the useful sensor is rarely
        named 'coretemp', so fall back to whatever thermal zone looks like a
        CPU, then to the hottest zone available.
        """
        try:
            temps = self.psutil.sensors_temperatures()
        except (AttributeError, OSError):
            return None
        if not temps:
            return None

        for key in ("coretemp", "k10temp", "cpu_thermal", "cpu-thermal",
                    "soc_thermal", "thermal-fan-est"):
            entries = temps.get(key)
            if entries:
                vals = [e.current for e in entries if e.current is not None]
                if vals:
                    return max(vals)

        vals = [e.current for group in temps.values() for e in group
                if e.current is not None and 0 < e.current < 150]
        return max(vals) if vals else None


def collect_disks() -> list:
    """
    Usage for each configured filesystem.

    A path may carry a display label after '=' -- "/host/disks/c=C:" -- which
    is how a Windows volume bind-mounted into the VM gets shown under the name
    its owner knows it by rather than as a mount point they never chose.
    """
    import psutil

    out = []
    for spec in config.DISK_PATHS:
        path, _, label = spec.partition("=")
        path = path.strip()
        if not path:
            continue
        try:
            du = psutil.disk_usage(path)
        except OSError:
            # A configured-but-absent mount is worth showing as such: silently
            # dropping it looks identical to never having configured it.
            out.append({"path": path, "label": (label.strip() or path),
                        "total": None, "used": None, "free": None,
                        "pct": None, "error": "unreadable"})
            continue
        out.append({
            "path": path,
            "label": label.strip() or path,
            "total": du.total,
            "used": du.used,
            "free": du.free,
            "pct": round(du.percent, 1),
            "error": None,
        })
    return out


def collect_host_procs(limit=12) -> list:
    """
    Heaviest processes by resident memory.

    This is the fallback view for hosts where the GPU cannot say who is
    holding its memory -- WSL2 being the case in hand. It answers a weaker
    question than the GPU process table (what is big, not what is on the card)
    and the UI labels it as such, but on a box running llama.cpp and ComfyUI
    the heavy processes are the interesting ones either way.
    """
    import psutil

    rows = []
    for proc in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
        try:
            info = proc.info
            mi = info.get("memory_info")
            if not mi:
                continue
            rows.append({
                "pid": info["pid"],
                "name": info.get("name") or "?",
                "rss": mi.rss,
                "cpu": info.get("cpu_percent") or 0.0,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    rows.sort(key=lambda r: r["rss"], reverse=True)
    return rows[:limit]


def capabilities() -> dict:
    """
    What this host can and cannot report.

    Stated up front so the frontend can distinguish "nothing is happening"
    from "this platform does not expose that", which look identical in the
    data and mean very different things to someone reading the page.
    """
    wsl = is_wsl()
    return {
        "wsl": wsl,
        "containerised": config.HOST_PROC_ACTIVE,
        # The WSL driver does not implement per-process GPU accounting, so the
        # query returns an empty list on a perfectly busy card.
        "gpu_process_accounting": bool(_NVIDIA_SMI) and not wsl,
        # No hwmon inside the WSL VM. GPU temperature still comes from the
        # driver and is real.
        "cpu_temperature": not wsl,
    }


def system_info() -> dict:
    global _BOOT_TIME
    import psutil

    if _BOOT_TIME is None:
        _BOOT_TIME = psutil.boot_time()

    gpus = collect_gpus()
    host_total = psutil.virtual_memory().total
    # A part that reports no framebuffer size has no framebuffer: on GB10 the
    # GPU draws from the host's DRAM, so that absence is the unified signal.
    unified = bool(gpus) and gpus[0].get("mem_total") is None
    reconcile_memory(gpus, collect_processes(), host_total)

    info = {
        "hostname": host_label(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "kernel": platform.release(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "boot_time": _BOOT_TIME,
        "mem_total": psutil.virtual_memory().total,
        "gpu_present": bool(gpus),
        "gpu_name": gpus[0]["name"] if gpus else None,
        "gpu_count": len(gpus),
        "gpu_mem_total": gpus[0]["mem_total"] if gpus else None,
        "gpu_power_limit": gpus[0].get("power_limit") if gpus else None,
        "disks": collect_disks(),
    }
    info.update(driver_info())
    info["capabilities"] = capabilities()

    # The other unified shape: the part does report a total, and it lands
    # within a few percent of system RAM because it is system RAM. Showing two
    # independent gauges there would double-count the same DRAM.
    #
    # Size alone is not enough evidence, though. A 12 GB discrete card on a VM
    # capped at 12 GB of RAM matches the same test and is emphatically not
    # unified -- which is a live possibility under WSL2, where the visible RAM
    # ceiling is whatever .wslconfig says. So the coincidence only counts when
    # the part's name also says it draws from system memory.
    if not unified:
        gmt = info.get("gpu_mem_total")
        name = (info.get("gpu_name") or "").lower()
        if gmt and any(hint in name for hint in _UNIFIED_HINTS):
            gpu_bytes = gmt * 1024 * 1024
            if host_total and abs(gpu_bytes - host_total) / host_total < 0.15:
                unified = True
    info["unified_memory"] = unified
    return info


def host_label() -> str:
    """
    The name to show for this machine.

    A container's own hostname is a random hex string, so an explicit label
    wins. Failing that, read the hostname from the mounted host procfs -- on
    WSL2 that is the Windows machine name, which is the name its owner
    actually calls it.
    """
    if config.HOST_LABEL:
        return config.HOST_LABEL
    if config.HOST_PROC_ACTIVE:
        try:
            with open(f"{config.HOST_PROC}/sys/kernel/hostname", "r",
                      encoding="utf-8", errors="replace") as fh:
                name = fh.read().strip()
            if name:
                return name
        except OSError:
            pass
    return socket.gethostname()


def collect(host: HostCollector, want_host_procs=False) -> dict:
    """One full sample: timestamp, host metrics, GPU rows, GPU processes."""
    host_metrics = host.collect()
    gpus = collect_gpus()
    procs = collect_processes()
    reconcile_memory(gpus, procs, host_metrics.get("mem_total"))
    sample = {
        "ts": time.time(),
        "host": host_metrics,
        "gpus": gpus,
        "procs": procs,
    }
    # Walking every /proc entry is far heavier than the rest of a sample, so
    # it runs on its own slower cadence rather than on every tick.
    if want_host_procs:
        sample["host_procs"] = collect_host_procs()
    return sample
