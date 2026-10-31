"""
avatar_maker.py - make a LAM avatar from an uploaded photo, over HTTP.

Runs export_avatar.py (one photo -> chat-avatar ZIP) for the avatar page, one
job at a time, and copies the finished ZIP into the folder the page serves.
Run it in the `lam` conda env (it uses that env's Python for the export).
From the repository root:

    conda activate lam
    uvicorn avatar_maker:app --app-dir avatar/scripts --host 0.0.0.0 --port 8703 --no-proxy-headers

--no-proxy-headers matters: without it, uvicorn trusts X-Forwarded-For from a
proxy on 127.0.0.1, and behind a CDN that header holds the visitor's public
address, which MAKER_ALLOW then refuses.

Settings (environment variables):
    LAM_DIR       LAM repo with export_avatar.py     default ~/LAM
    BLENDER       Blender 4.x executable             default ~/software/blender-4.0.2-linux-x64/blender
    AVATAR_DIRS   where finished ZIPs go; several    default avatar/html/avatars next
                  folders separated by ':'           to this script
    MAKER_ALLOW   client networks that may call the  default 127.0.0.0/8,::1/128,
                  maker directly, comma-separated    172.16.0.0/12 (Docker)

Endpoints
    GET  /health                      -> {"ok": true, "busy": bool, "queued": n}
    POST /jobs?name=josh&replace=0    body = the photo (JPEG, PNG, WebP, HEIC if
                                      Pillow can read it), at most 15 MB
                                      -> 202 {"id", "name", "state": "queued", ...}
    GET  /jobs/{id}                   -> {"id", "name", "state": queued|running|done|error,
                                          "step": 0-5, "steps": 5, "message",
                                          "error", "avatar": "avatars/<name>.zip",
                                          "elapsed": s}
    GET  /jobs/{id}/preview           -> PNG of the face crop LAM used (when done)
"""
import io
import ipaddress
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

HOME = Path.home()
LAM_DIR = Path(os.environ.get("LAM_DIR", HOME / "LAM")).expanduser()
BLENDER = Path(os.environ.get("BLENDER", HOME / "software/blender-4.0.2-linux-x64/blender")).expanduser()
DEFAULT_AVATARS = Path(__file__).resolve().parent.parent / "html/avatars"
AVATAR_DIRS = [Path(p).expanduser() for p in
               os.environ.get("AVATAR_DIRS", str(DEFAULT_AVATARS)).split(":") if p]

# Who may call the maker directly. It listens on all interfaces so the avatar
# container can reach it, but outside callers must come through nginx (which
# has the LAN-only rule): the container's requests come from Docker's range.
ALLOW = [ipaddress.ip_network(n) for n in
         os.environ.get("MAKER_ALLOW", "127.0.0.0/8,::1/128,172.16.0.0/12").split(",") if n]

MAX_BYTES = 15 * 1024 * 1024
MAX_SIDE = 2048            # bigger photos are scaled down; LAM crops to the face anyway
MAX_QUEUE = 3
TIMEOUT_S = 15 * 60
KEEP_S = 60 * 60           # finished jobs are forgotten after an hour
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
STEP_RE = re.compile(r"^\[(\d)/(\d)\]\s*(.*)")
REQUIRED = ("offset.ply", "skin.glb", "animation.glb", "vertex_order.json")

WORK = Path(tempfile.mkdtemp(prefix="avatar_maker_"))
jobs: dict[str, dict] = {}
queue: deque[str] = deque()
lock = threading.Lock()
wake = threading.Event()


def public(job: dict) -> dict:
    out = {k: job[k] for k in ("id", "name", "state", "step", "steps", "message", "error", "avatar")}
    end = job["finished"] or time.time()
    out["elapsed"] = round(end - job["started"]) if job["started"] else 0
    if job["state"] == "queued":
        out["position"] = list(queue).index(job["id"]) + 1 if job["id"] in queue else 0
    return out


def explain(log: list[str], code: int, step: int) -> str:
    text = "\n".join(log)
    if "out of memory" in text.lower():
        return ("The GPU ran out of memory. Making an avatar needs about 4.5 GB free; "
                "try again when the GPU is less busy.")
    if step == 2:   # FLAME fitting: LAM couldn't find or fit a face
        return ("Couldn't find a face to fit in that photo. Use a well-lit, front-facing "
                "photo with the whole head in frame and nothing covering the face.")
    for line in reversed(log):
        if "AssertionError" in line or line.startswith(("Missing ", "Run this from")):
            return line.split("AssertionError:")[-1].strip() or line
    for line in reversed(log):
        if line.strip() and not line.startswith(" "):
            return f"Export failed: {line.strip()[:300]}"
    return f"Export failed (exit code {code})."


def run(job: dict) -> None:
    jdir = WORK / job["id"]
    tmp_name = f"maker_{job['id']}"          # LAM's own output folder for this run
    out_root = LAM_DIR / "output/open_avatar_chat"
    cmd = [sys.executable, "-u", "export_avatar.py", str(jdir / "photo.png"),
           "--blender", str(BLENDER), "--name", tmp_name]
    # Blender resolves relative paths (LAM passes "--python tools/...") against
    # $PWD, not the real working directory, and exits 0 when the script is
    # missing. So PWD has to match cwd, or skin.glb is silently never made.
    env = {**os.environ, "PWD": str(LAM_DIR), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    log: list[str] = []
    proc = subprocess.Popen(cmd, cwd=LAM_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    killer = threading.Timer(TIMEOUT_S, proc.kill)
    killer.start()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            log.append(line)
            del log[:-200]
            m = STEP_RE.match(line)
            if m:
                job["step"] = int(m.group(1))
                job["message"] = m.group(3).replace(tmp_name, job["name"])
        code = proc.wait()
    finally:
        killer.cancel()
        (jdir / "export.log").write_text("\n".join(log))

    try:
        zip_path = out_root / f"{tmp_name}.zip"
        if code != 0 or not zip_path.exists():
            if code == -9:
                raise RuntimeError(f"Export took longer than {TIMEOUT_S // 60} minutes and was stopped.")
            raise RuntimeError(explain(log, code, job["step"]))
        with zipfile.ZipFile(zip_path) as z:
            have = {Path(n).name for n in z.namelist()}
        missing = [f for f in REQUIRED if f not in have]
        if missing:
            raise RuntimeError("The export finished without " + ", ".join(missing) +
                               ". Blender probably failed; see the maker's log.")
        # The export ran under a temporary name so it never touches LAM output
        # you made yourself; repack it with the real name as the top folder.
        packed = jdir / "avatar.zip"
        with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                new = job["name"] + info.filename[len(tmp_name):] \
                    if info.filename.startswith(tmp_name) else info.filename
                dst.writestr(zipfile.ZipInfo(new, info.date_time), src.read(info),
                             compress_type=zipfile.ZIP_DEFLATED)
        for d in AVATAR_DIRS:
            d.mkdir(parents=True, exist_ok=True)
            part = d / f".{job['name']}.zip.part"
            shutil.copyfile(packed, part)
            os.replace(part, d / f"{job['name']}.zip")
        crop = out_root / f"{tmp_name}_processed.png"
        if crop.exists():
            shutil.copyfile(crop, jdir / "preview.png")
        job.update(state="done", step=5, message="Done", avatar=f"avatars/{job['name']}.zip")
    except Exception as e:
        job.update(state="error", error=str(e), message="Failed")
        print(f"[maker] {job['id']} failed:\n" + "\n".join(log[-40:]), flush=True)
    finally:
        shutil.rmtree(out_root / tmp_name, ignore_errors=True)
        for f in (f"{tmp_name}.zip", f"{tmp_name}_processed.png"):
            (out_root / f).unlink(missing_ok=True)


def worker() -> None:
    while True:
        wake.wait()
        with lock:
            job_id = queue.popleft() if queue else None
            if not queue:
                wake.clear()
        if not job_id:
            continue
        job = jobs[job_id]
        job.update(state="running", started=time.time(), message="Starting")
        print(f"[maker] {job_id} '{job['name']}' started", flush=True)
        try:
            run(job)
        except Exception as e:  # never let one job kill the worker
            job.update(state="error", error=f"Maker error: {e}", message="Failed")
        job["finished"] = time.time()
        print(f"[maker] {job_id} '{job['name']}' {job['state']} "
              f"in {job['finished'] - job['started']:.0f}s", flush=True)


def prune() -> None:
    now = time.time()
    for job_id, job in list(jobs.items()):
        if job["finished"] and now - job["finished"] > KEEP_S:
            shutil.rmtree(WORK / job_id, ignore_errors=True)
            jobs.pop(job_id, None)


app = FastAPI(title="LAM avatar maker")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def only_local(request: Request, call_next):
    try:
        ip = ipaddress.ip_address(request.client.host)
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        allowed = any(ip in net for net in ALLOW)
    except (ValueError, AttributeError):
        allowed = False
    if not allowed:
        await drain(request)
        return JSONResponse({"detail": "Use the avatar page to make avatars."}, status_code=403)
    return await call_next(request)


async def drain(request: Request) -> None:
    """Read (and drop) an upload before answering. Replying while a proxy is
    still sending the photo makes the connection reset, and nginx then shows
    the user a bare 502 instead of the real error."""
    seen = 0
    async for chunk in request.stream():
        seen += len(chunk)
        if seen > 2 * MAX_BYTES:
            break


@app.on_event("startup")
def start() -> None:
    for path, what in [(LAM_DIR / "export_avatar.py", "export_avatar.py in LAM_DIR"),
                       (BLENDER, "Blender (set BLENDER)")]:
        if not path.exists():
            raise RuntimeError(f"Missing {what}: {path}")
    threading.Thread(target=worker, daemon=True).start()
    print(f"[maker] ready. LAM={LAM_DIR} output={':'.join(map(str, AVATAR_DIRS))}", flush=True)


@app.get("/health")
def health():
    busy = any(j["state"] == "running" for j in jobs.values())
    return {"ok": True, "busy": busy, "queued": len(queue)}


@app.post("/jobs", status_code=202)
async def create(request: Request, name: str, replace: bool = False):
    # Read the whole upload before any check can fail (see drain()).
    body = b""
    async for chunk in request.stream():
        if len(body) <= MAX_BYTES:
            body += chunk
    name = name.strip().lower()
    if not NAME_RE.match(name):
        raise HTTPException(400, "Use a name of 1–40 letters, digits, '-' or '_', starting with a letter or digit.")
    if not replace and any((d / f"{name}.zip").exists() for d in AVATAR_DIRS):
        raise HTTPException(409, f"An avatar called '{name}' already exists.")
    if not body:
        raise HTTPException(400, "Send the photo as the request body.")
    if len(body) > MAX_BYTES:
        raise HTTPException(413, "The photo is larger than 15 MB.")
    try:
        with Image.open(io.BytesIO(body)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")   # phone photos are often stored sideways
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            prune()
            with lock:
                if len(queue) >= MAX_QUEUE:
                    raise HTTPException(429, "Too many avatars waiting. Try again in a few minutes.")
                if any(j["name"] == name and j["state"] in ("queued", "running") for j in jobs.values()):
                    raise HTTPException(409, f"'{name}' is already being made.")
                job_id = uuid.uuid4().hex[:12]
                (WORK / job_id).mkdir()
                im.save(WORK / job_id / "photo.png")
                jobs[job_id] = {"id": job_id, "name": name, "state": "queued", "step": 0, "steps": 5,
                                "message": "Waiting", "error": None, "avatar": None,
                                "started": None, "finished": None}
                queue.append(job_id)
                wake.set()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(415, "That file isn't a photo this server can read. Use a JPEG or PNG.")
    return JSONResponse(public(jobs[job_id]), status_code=202)


@app.get("/jobs/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "No such job (finished jobs are kept for an hour).")
    return public(job)


@app.get("/jobs/{job_id}/preview")
def preview(job_id: str):
    path = WORK / job_id / "preview.png"
    if job_id not in jobs or not path.exists():
        raise HTTPException(404, "No preview for this job.")
    return FileResponse(path, media_type="image/png")
