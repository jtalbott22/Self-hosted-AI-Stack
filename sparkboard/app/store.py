"""
Time-series storage for Sparkboard.

Two tiers, so that a 30-day view doesn't mean scanning a million rows:

  host / gpu        raw samples at the collection interval, kept ~26h
  host_1m / gpu_1m  one-minute rollups, kept ~120d

The rollup job simply re-aggregates the last few minutes on every pass with
INSERT OR REPLACE. That's idempotent and self-healing -- no bookkeeping table
to get out of sync, and a restart mid-minute can't leave a hole.

Queries bucket on read (GROUP BY ts/bucket) and return roughly 700 points
regardless of the range asked for, so the payload stays small whether the
window is five minutes or a month.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time

log = logging.getLogger("sparkboard.store")

from . import config

# Retention is configurable because the two hosts this runs on have very
# different disks. The defaults are the bare-metal ones.
RAW_RETENTION = int(config.RAW_RETENTION_HOURS * 3600)
ROLL_RETENTION = int(config.RETENTION_DAYS * 86400)
ROLLUP_WINDOW = 600                # re-aggregate this much history each pass
TARGET_POINTS = 700                # approximate points returned per query

HOST_COLS = [
    "cpu", "cpu_temp", "mem_used", "mem_total", "swap_used", "load1",
    "net_rx", "net_tx", "disk_read", "disk_write",
]
GPU_COLS = [
    "util", "mem_util", "mem_used", "mem_total", "temp", "power",
    "sm_clock", "mem_clock", "fan",
]
# Per-container series live in a long-format table keyed by name, the same
# shape the GPU table uses for its index. No dynamic columns needed.
CONTAINER_COLS = [
    "cpu", "mem_used", "mem_pct",
    "net_rx_rate", "net_tx_rate", "block_read_rate", "block_write_rate",
]
# Columns where a bucket's peak matters as much as its mean: at a one-hour
# bucket an averaged thermal spike disappears entirely.
HOST_MAX_COLS = ["cpu", "cpu_temp"]
GPU_MAX_COLS = ["util", "temp", "power"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS host (
  ts INTEGER PRIMARY KEY,
  {host_defs}
);
CREATE TABLE IF NOT EXISTS host_1m (
  ts INTEGER PRIMARY KEY,
  {host_defs}
);
CREATE TABLE IF NOT EXISTS gpu (
  ts INTEGER NOT NULL,
  idx INTEGER NOT NULL,
  {gpu_defs},
  PRIMARY KEY (ts, idx)
);
CREATE TABLE IF NOT EXISTS gpu_1m (
  ts INTEGER NOT NULL,
  idx INTEGER NOT NULL,
  {gpu_defs},
  PRIMARY KEY (ts, idx)
);
CREATE TABLE IF NOT EXISTS container (
  ts INTEGER NOT NULL,
  name TEXT NOT NULL,
  {container_defs},
  PRIMARY KEY (ts, name)
);
CREATE TABLE IF NOT EXISTS container_1m (
  ts INTEGER NOT NULL,
  name TEXT NOT NULL,
  {container_defs},
  PRIMARY KEY (ts, name)
);
CREATE INDEX IF NOT EXISTS gpu_ts ON gpu(ts);
CREATE INDEX IF NOT EXISTS gpu_1m_ts ON gpu_1m(ts);
CREATE INDEX IF NOT EXISTS container_ts ON container(ts);
CREATE INDEX IF NOT EXISTS container_1m_ts ON container_1m(ts);
""".format(
    host_defs=",\n  ".join(f"{c} REAL" for c in HOST_COLS),
    gpu_defs=",\n  ".join(f"{c} REAL" for c in GPU_COLS),
    container_defs=",\n  ".join(f"{c} REAL" for c in CONTAINER_COLS),
)


class Store:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL lets the HTTP handlers read while the sampler thread writes.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- write

    def insert(self, sample: dict):
        ts = int(sample["ts"])
        host = sample.get("host") or {}
        gpus = sample.get("gpus") or []

        host_vals = [ts] + [host.get(c) for c in HOST_COLS]
        host_sql = (
            f"INSERT OR REPLACE INTO host (ts, {', '.join(HOST_COLS)}) "
            f"VALUES ({', '.join('?' * (len(HOST_COLS) + 1))})"
        )
        gpu_sql = (
            f"INSERT OR REPLACE INTO gpu (ts, idx, {', '.join(GPU_COLS)}) "
            f"VALUES ({', '.join('?' * (len(GPU_COLS) + 2))})"
        )
        gpu_rows = [
            [ts, g.get("index", 0)] + [g.get(c) for c in GPU_COLS]
            for g in gpus
        ]

        with self._lock:
            self._conn.execute(host_sql, host_vals)
            if gpu_rows:
                self._conn.executemany(gpu_sql, gpu_rows)
            self._conn.commit()

    def insert_containers(self, ts: float, containers: list):
        if not containers:
            return
        ts = int(ts)
        sql = (
            f"INSERT OR REPLACE INTO container (ts, name, {', '.join(CONTAINER_COLS)}) "
            f"VALUES ({', '.join('?' * (len(CONTAINER_COLS) + 2))})"
        )
        rows = [
            [ts, c.get("name") or "?"] + [c.get(col) for col in CONTAINER_COLS]
            for c in containers
        ]
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    # ------------------------------------------------------------- maintain

    def _rollup_since(self, since: int):
        host_aggs = ", ".join(f"AVG({c})" for c in HOST_COLS)
        gpu_aggs = ", ".join(f"AVG({c})" for c in GPU_COLS)
        cont_aggs = ", ".join(f"AVG({c})" for c in CONTAINER_COLS)

        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO host_1m (ts, {', '.join(HOST_COLS)}) "
                f"SELECT (ts/60)*60, {host_aggs} FROM host "
                f"WHERE ts >= ? GROUP BY ts/60",
                (since,),
            )
            self._conn.execute(
                f"INSERT OR REPLACE INTO gpu_1m (ts, idx, {', '.join(GPU_COLS)}) "
                f"SELECT (ts/60)*60, idx, {gpu_aggs} FROM gpu "
                f"WHERE ts >= ? AND idx IS NOT NULL GROUP BY ts/60, idx",
                (since,),
            )
            self._conn.execute(
                f"INSERT OR REPLACE INTO container_1m (ts, name, {', '.join(CONTAINER_COLS)}) "
                f"SELECT (ts/60)*60, name, {cont_aggs} FROM container "
                f"WHERE ts >= ? GROUP BY ts/60, name",
                (since,),
            )
            self._conn.commit()

    def rollup(self):
        """Steady-state pass: re-aggregate the last few minutes."""
        self._rollup_since(int(time.time()) - ROLLUP_WINDOW)

    def backfill(self):
        """
        Close any gap between the raw and rollup tiers.

        Called once at startup. Without this, raw samples written while the
        rollup job wasn't running -- a first install, or any restart after
        more than ROLLUP_WINDOW of downtime -- would sit in the raw table and
        never reach the one-minute tier, so they'd be missing from the 7d and
        30d views despite being on disk.
        """
        with self._lock:
            row = self._conn.execute("SELECT MAX(ts) FROM host_1m").fetchone()
            last_rolled = row[0] if row and row[0] is not None else None
            row = self._conn.execute("SELECT MIN(ts) FROM host").fetchone()
            oldest_raw = row[0] if row and row[0] is not None else None

        if oldest_raw is None:
            return 0
        # Redo the final rolled-up minute: it may have been written from a
        # partial minute of samples.
        since = oldest_raw if last_rolled is None else min(last_rolled, oldest_raw)
        self._rollup_since(int(since))

        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM host_1m").fetchone()
        return row[0] if row else 0

    def prune(self):
        now = int(time.time())
        with self._lock:
            self._conn.execute("DELETE FROM host WHERE ts < ?", (now - RAW_RETENTION,))
            self._conn.execute("DELETE FROM gpu WHERE ts < ?", (now - RAW_RETENTION,))
            self._conn.execute("DELETE FROM host_1m WHERE ts < ?", (now - ROLL_RETENTION,))
            self._conn.execute("DELETE FROM gpu_1m WHERE ts < ?", (now - ROLL_RETENTION,))
            self._conn.execute("DELETE FROM container WHERE ts < ?", (now - RAW_RETENTION,))
            self._conn.execute("DELETE FROM container_1m WHERE ts < ?", (now - ROLL_RETENTION,))
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            def one(sql):
                r = self._conn.execute(sql).fetchone()
                return r[0] if r else None
            out = {
                "raw_rows": one("SELECT COUNT(*) FROM host"),
                "rollup_rows": one("SELECT COUNT(*) FROM host_1m"),
                "oldest_raw": one("SELECT MIN(ts) FROM host"),
                "oldest_rollup": one("SELECT MIN(ts) FROM host_1m"),
            }
        try:
            out["db_bytes"] = os.path.getsize(self.path)
        except OSError:
            out["db_bytes"] = None
        return out

    def container_history(self, range_s: int, base_interval: float) -> dict:
        """
        Per-container series on a shared timeline.

        Containers that started or stopped inside the window simply have nulls
        outside their lifetime, which the sparklines draw as gaps -- so a
        container that was restarted looks restarted rather than looking like
        it idled at zero.
        """
        suffix, bucket = self.plan(range_s, base_interval)
        now = int(time.time())
        since = now - range_s

        sel = ", ".join(f"AVG({c}) AS {c}" for c in CONTAINER_COLS)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT (ts/{bucket})*{bucket} AS b, name, {sel} "
                f"FROM container{suffix} WHERE ts >= ? "
                f"GROUP BY b, name ORDER BY b",
                (since,),
            ).fetchall()

        timeline = sorted({r["b"] for r in rows})
        pos = {b: i for i, b in enumerate(timeline)}
        n = len(timeline)
        names = sorted({r["name"] for r in rows})

        out = {name: {c: [None] * n for c in CONTAINER_COLS} for name in names}
        for r in rows:
            i = pos[r["b"]]
            target = out[r["name"]]
            for c in CONTAINER_COLS:
                target[c][i] = r[c]

        return {
            "t": timeline,
            "containers": out,
            "bucket": bucket,
            "range": range_s,
            "tier": "1m" if suffix else "raw",
            "points": n,
        }

    # ----------------------------------------------------------------- read

    @staticmethod
    def plan(range_s: int, base_interval: float):
        """Pick the source tier and bucket width for a requested window."""
        if range_s <= 24 * 3600:
            table_suffix = ""
            min_bucket = max(1, int(base_interval))
        else:
            table_suffix = "_1m"
            min_bucket = 60
        bucket = max(min_bucket, int(range_s / TARGET_POINTS))
        # Round to a tidy interval so bucket edges land on clean clock times.
        for nice in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900,
                     1800, 3600, 7200, 14400, 21600, 43200, 86400):
            if nice >= bucket:
                bucket = nice
                break
        return table_suffix, bucket

    def history(self, range_s: int, base_interval: float, gpu_idx: int = 0) -> dict:
        suffix, bucket = self.plan(range_s, base_interval)
        now = int(time.time())
        since = now - range_s

        host_sel = ", ".join(f"AVG({c}) AS {c}" for c in HOST_COLS)
        host_sel += ", " + ", ".join(f"MAX({c}) AS {c}_max" for c in HOST_MAX_COLS)
        gpu_sel = ", ".join(f"AVG({c}) AS {c}" for c in GPU_COLS)
        gpu_sel += ", " + ", ".join(f"MAX({c}) AS {c}_max" for c in GPU_MAX_COLS)

        with self._lock:
            host_rows = self._conn.execute(
                f"SELECT (ts/{bucket})*{bucket} AS b, {host_sel} "
                f"FROM host{suffix} WHERE ts >= ? "
                f"GROUP BY b ORDER BY b",
                (since,),
            ).fetchall()
            gpu_rows = self._conn.execute(
                f"SELECT (ts/{bucket})*{bucket} AS b, {gpu_sel} "
                f"FROM gpu{suffix} WHERE ts >= ? AND idx = ? "
                f"GROUP BY b ORDER BY b",
                (since, gpu_idx),
            ).fetchall()

        # Merge both tables onto one shared timeline. A bucket present in only
        # one table still gets a slot; the other side is null there, and the
        # chart draws a gap rather than interpolating across missing data.
        timeline = sorted({r["b"] for r in host_rows} | {r["b"] for r in gpu_rows})
        pos = {b: i for i, b in enumerate(timeline)}
        n = len(timeline)

        series = {}
        host_keys = HOST_COLS + [f"{c}_max" for c in HOST_MAX_COLS]
        gpu_keys = GPU_COLS + [f"{c}_max" for c in GPU_MAX_COLS]
        for k in host_keys:
            series[f"host_{k}"] = [None] * n
        for k in gpu_keys:
            series[f"gpu_{k}"] = [None] * n

        for r in host_rows:
            i = pos[r["b"]]
            for k in host_keys:
                series[f"host_{k}"][i] = r[k]
        for r in gpu_rows:
            i = pos[r["b"]]
            for k in gpu_keys:
                series[f"gpu_{k}"][i] = r[k]

        return {
            "t": timeline,
            "series": series,
            "bucket": bucket,
            "range": range_s,
            "tier": "1m" if suffix else "raw",
            "points": n,
        }
