"""Music for the avatar page: search and stream songs, radio and podcasts.

    GET /health
    GET /search?q=...&kind=music|radio|podcast   -> {"items": [...]}
    GET /stream/{token}                          -> the audio, same origin

Sources, none of which need an account or a key:
    music    YouTube, through yt-dlp. The first hit is followed by its YouTube
             "mix", so a song keeps going with related songs.
    radio    radio-browser.info, the open directory of internet radio.
    podcast  Apple's podcast search for the feed, then the feed's RSS.

All audio goes through /stream, because:
  - YouTube's audio URLs only work from the address that asked for them
    (this server), not from a phone on another network;
  - most radio streams are plain http, which an https page can't play;
  - one origin keeps the page simple and works in the car.
Search results hand out short tokens instead of URLs, so /stream can only
fetch what a search returned, not any address a caller makes up.
"""

import asyncio
import hashlib
import os
import re
import secrets
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
import numpy as np
import yt_dlp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

UA = "avatar-music/1.0 (self-hosted voice assistant)"
RADIO_API = os.environ.get("RADIO_API", "https://all.api.radio-browser.info")
TOKEN_TTL = 12 * 3600            # how long a search result stays playable
CACHE = Path(os.environ.get("CACHE_DIR", "/cache"))
CACHE_BYTES = int(float(os.environ.get("CACHE_GB", "2")) * 1024**3)
MIX_LEN = 25

app = FastAPI(title="avatar music")
CACHE.mkdir(parents=True, exist_ok=True)
http = httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True,
                         timeout=httpx.Timeout(20, read=60))

# token -> {"kind", "src", "exp"}
TOKENS: dict[str, dict] = {}


def remember(kind: str, src: str) -> str:
    now = time.time()
    for t in [t for t, v in TOKENS.items() if v["exp"] < now]:
        del TOKENS[t]
    tok = secrets.token_urlsafe(12)
    TOKENS[tok] = {"kind": kind, "src": src, "exp": now + TOKEN_TTL}
    return tok


def item(kind, src, title, subtitle="", art="", duration=None, live=False,
         link="", link_label="", credit="", credit_link=""):
    """One search result. link/credit say where it came from, for the page to cite."""
    tok = remember(kind, src)
    return {"kind": kind, "title": title or "Untitled", "subtitle": subtitle or "",
            "art": art or "", "duration": duration, "live": live,
            "url": "stream/" + tok, "levels": None if live else "levels/" + tok,
            "link": link or "", "linkLabel": link_label or "",
            "credit": credit or "", "creditLink": credit_link or ""}


# --------------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------------
YDL_FLAT = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist", "noplaylist": False}
# A mix is endless; stop paging once there's enough (paging all of it takes ~20 s).
YDL_MIX = {**YDL_FLAT, "playlistend": MIX_LEN + 1}
# m4a (AAC) first: it plays everywhere, Safari and CarPlay included.
YDL_AUDIO = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
             "format": "bestaudio[ext=m4a]/bestaudio[acodec^=mp4a]/bestaudio/best",
             "outtmpl": "%(id)s.%(ext)s",
             # ffmpeg turns YouTube's DASH m4a into a plain one iPhones can seek in
             "fixup": "force"}

def yt_thumb(e):
    vid = e.get("id")
    return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else ""


def yt_entry(e):
    who = e.get("channel") or e.get("uploader") or ""
    return item("music", e["id"], e.get("title"), who, yt_thumb(e), e.get("duration"),
                link=f"https://www.youtube.com/watch?v={e['id']}", link_label="Watch on YouTube",
                credit=f"Uploaded to YouTube by {who}" if who else "From YouTube",
                credit_link=f"https://www.youtube.com/channel/{e['channel_id']}" if e.get("channel_id") else "")


def yt_search(q: str):
    with yt_dlp.YoutubeDL(YDL_FLAT) as y:
        res = y.extract_info(f"ytsearch8:{q}", download=False) or {}
    hits = [e for e in res.get("entries") or []
            if e and e.get("id") and (e.get("duration") or 0) < 3 * 3600
            and "/shorts/" not in (e.get("url") or "")]
    if not hits:
        return []
    first = hits[0]
    # The first hit's mix: related songs, the way YouTube Music keeps playing.
    try:
        with yt_dlp.YoutubeDL(YDL_MIX) as y:
            mix = y.extract_info(
                f"https://www.youtube.com/watch?v={first['id']}&list=RD{first['id']}",
                download=False) or {}
        rest = [e for e in mix.get("entries") or [] if e and e.get("id") and e["id"] != first["id"]]
    except Exception:
        rest = []
    if len(rest) < 3:
        rest = hits[1:]
    return [yt_entry(first)] + [yt_entry(e) for e in rest[:MIX_LEN]]


def yt_fetch(vid: str) -> Path:
    """Download one song into the cache, once. Songs are a few MB, so this
    takes a second or two; the page asks for the next one ahead of time."""
    done = CACHE / f"{vid}.m4a"
    if done.exists():
        os.utime(done)
        return done
    work = CACHE / f"tmp-{vid}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        with yt_dlp.YoutubeDL({**YDL_AUDIO, "paths": {"home": str(work)}}) as y:
            y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
        got = [f for f in work.iterdir() if f.is_file() and not f.name.endswith(".part")]
        if not got:
            raise yt_dlp.utils.DownloadError("nothing was downloaded")
        src = got[0]
        if src.suffix != ".m4a":                 # no AAC version: convert, iPhone can't play opus/webm everywhere
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vn",
                            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                            str(work / "out.m4a")], check=True, timeout=600)
            src = work / "out.m4a"
        os.replace(src, done)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    try:
        compute_levels(str(done), LEVELS / f"{vid}.lv")      # about a second for a song
    except Exception:
        pass
    prune()
    return done


def prune():
    files = sorted((f for f in CACHE.glob("*.m4a")), key=lambda f: f.stat().st_mtime, reverse=True)
    total = 0
    for f in files:
        total += f.stat().st_size
        if total > CACHE_BYTES:
            f.unlink(missing_ok=True)
            (LEVELS / f"{f.stem}.lv").unlink(missing_ok=True)
    # podcast levels are small; keep the newest few hundred
    for f in sorted(LEVELS.glob("*.lv"), key=lambda f: f.stat().st_mtime, reverse=True)[400:]:
        f.unlink(missing_ok=True)


FETCHING: dict[str, asyncio.Task] = {}


async def yt_file(vid: str) -> Path:
    """One download per song, however many requests ask for it at once."""
    task = FETCHING.get(vid)
    if task is None:
        task = asyncio.create_task(asyncio.to_thread(yt_fetch, vid))
        FETCHING[vid] = task
        task.add_done_callback(lambda _t: FETCHING.pop(vid, None))
    return await asyncio.shield(task)


# --------------------------------------------------------------------------
# Radio
# --------------------------------------------------------------------------
PLAYABLE = {"MP3", "AAC", "AAC+", "OGG"}


async def radio_search(q: str):
    params = {"limit": 30, "hidebroken": "true", "order": "clickcount", "reverse": "true"}
    found = []
    for field in ("name", "tag"):                 # "WNYC", then "jazz"
        r = await http.get(f"{RADIO_API}/json/stations/search", params={**params, field: q})
        r.raise_for_status()
        found = [s for s in r.json()
                 if s.get("url_resolved") and not s.get("hls")
                 and (s.get("codec") or "").upper() in PLAYABLE]
        if found:
            break
    out, seen = [], set()
    for s in found:
        key = s["name"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        where = ", ".join(x for x in (s.get("state"), s.get("country")) if x)
        tags = ", ".join((s.get("tags") or "").split(",")[:3])
        out.append(item("radio", s["url_resolved"], s["name"].strip(),
                        " · ".join(x for x in (tags, where) if x),
                        s.get("favicon") or "", None, True,
                        link=s.get("homepage") or "", link_label="Station website",
                        credit="Stream listed on radio-browser.info",
                        credit_link="https://www.radio-browser.info/"))
        if len(out) >= 8:
            break
    return out


# --------------------------------------------------------------------------
# Podcasts
# --------------------------------------------------------------------------
ITUNES = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"


def seconds(v):
    if not v:
        return None
    try:
        parts = [int(float(p)) for p in v.strip().split(":")]
    except ValueError:
        return None
    n = 0
    for p in parts:
        n = n * 60 + p
    return n


async def podcast_search(q: str, episode: str = ""):
    r = await http.get("https://itunes.apple.com/search",
                       params={"media": "podcast", "entity": "podcast", "term": q, "limit": 5})
    r.raise_for_status()
    shows = [s for s in r.json().get("results", []) if s.get("feedUrl")]
    if not shows:
        return []
    show = shows[0]
    feed = await http.get(show["feedUrl"])
    feed.raise_for_status()
    root = ET.fromstring(feed.content)
    chan = root.find("channel")
    if chan is None:
        return []
    show_art = show.get("artworkUrl600") or show.get("artworkUrl100") or ""
    eps = []
    for it in chan.findall("item"):
        enc = it.find("enclosure")
        if enc is None or not enc.get("url"):
            continue
        if enc.get("type") and not enc.get("type", "").startswith("audio"):
            continue
        title = (it.findtext("title") or "").strip()
        when = ""
        try:
            when = parsedate_to_datetime(it.findtext("pubDate")).strftime("%b %-d, %Y")
        except Exception:
            pass
        img = it.find(ITUNES + "image")
        eps.append((title, enc.get("url"), when, img.get("href") if img is not None else "",
                    seconds(it.findtext(ITUNES + "duration")), (it.findtext("link") or "").strip()))
    if episode:                                  # "the one about ..." / an episode number
        words = [w for w in re.findall(r"\w+", episode.lower()) if len(w) > 1]
        scored = [(sum(w in e[0].lower() for w in words), -i, e) for i, e in enumerate(eps)]
        best = max(scored, default=None)
        if best and best[0] > 0:
            i = eps.index(best[2])
            eps = eps[i:]
    name = show.get("collectionName") or ""
    apple = show.get("collectionViewUrl") or ""
    by = show.get("artistName") or name
    return [item("podcast", url, title, " · ".join(x for x in (name, when) if x),
                 img or show_art, dur,
                 link=page if page.startswith("http") else apple,
                 link_label="Episode page" if page.startswith("http") else "Apple Podcasts",
                 credit=f"{name} by {by}, from the show's RSS feed" if by != name else f"{name}, from the show's RSS feed",
                 credit_link=apple)
            for title, url, when, img, dur, page in eps[:15]]


# --------------------------------------------------------------------------
# Levels for the page's visualizer
#
# The page can't analyse the music itself on an iPhone: that means routing
# the <audio> element through Web Audio, and iOS stops that when the screen
# locks. So each file is decoded here once, into a spectrum per frame
# (FPS frames a second, BANDS bands, one byte each), and the page plays the
# frames back in step with the music. That's about 0.5 KB per second of audio.
# --------------------------------------------------------------------------
FPS, BANDS, SR, WIN = 20, 24, 11025, 1024
HOP = SR // FPS
LEVELS = CACHE / "levels"
LEVELS.mkdir(parents=True, exist_ok=True)
_edges = np.geomspace(55, 5000, BANDS + 1) * WIN / SR
_bins = [(int(a), max(int(a) + 1, int(b))) for a, b in zip(_edges[:-1], _edges[1:])]
_hann = np.hanning(WIN).astype(np.float32)


def compute_levels(src: str, out: Path):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", src, "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "s16le", "-"], stdout=subprocess.PIPE)
    frames, tail = [], np.zeros(WIN - HOP, np.float32)
    try:
        while True:
            raw = p.stdout.read(HOP * 2 * 400)
            if not raw:
                break
            x = np.concatenate([tail, np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768])
            n = (len(x) - WIN) // HOP + 1
            if n <= 0:
                tail = x
                continue
            idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
            spec = np.abs(np.fft.rfft(x[idx] * _hann, axis=1)) ** 2
            frames.append(np.stack([spec[:, a:b].mean(axis=1) for a, b in _bins], axis=1))
            tail = x[n * HOP:]
    finally:
        p.stdout.close()
        p.wait(timeout=60)
    if not frames:
        raise RuntimeError("no audio")
    db = 10 * np.log10(np.concatenate(frames) + 1e-10)
    # Each band against its own range, so the highs move too, mixed with the
    # overall level, so quiet passages still look quiet.
    lo, hi = np.percentile(db, 8, axis=0), np.percentile(db, 99.5, axis=0)
    per = np.clip((db - lo) / np.maximum(hi - lo, 1), 0, 1)
    peak = np.percentile(db, 99.5)
    whole = np.clip((db - (peak - 55)) / 55, 0, 1)
    v = (0.6 * per + 0.4 * whole) ** 1.4
    tmp = out.with_suffix(".part")
    tmp.write_bytes((v * 255).astype(np.uint8).tobytes())
    os.replace(tmp, out)


LEVEL_TASKS: dict[str, asyncio.Task] = {}


def levels_key(t) -> str:
    return t["src"] if t["kind"] == "music" else hashlib.sha1(t["src"].encode()).hexdigest()


@app.get("/levels/{tok}")
async def levels(tok: str):
    """200 with the frames, or 202 while they're being worked out (ask again)."""
    t = TOKENS.get(tok)
    if not t or t["kind"] == "radio":
        raise HTTPException(404, "No levels for that.")
    key = levels_key(t)
    out = LEVELS / f"{key}.lv"
    hdr = {"x-fps": str(FPS), "x-bands": str(BANDS), "cache-control": "no-store"}
    if out.exists():
        return FileResponse(out, media_type="application/octet-stream", headers=hdr)
    if t["kind"] == "music":
        src = CACHE / f"{key}.m4a"
        if not src.exists():                     # the song itself isn't here yet
            return Response(status_code=202, headers=hdr)
        src = str(src)
    else:
        src = t["src"]                           # ffmpeg reads the podcast straight from its host
    if key not in LEVEL_TASKS:
        task = asyncio.create_task(asyncio.to_thread(compute_levels, src, out))
        LEVEL_TASKS[key] = task
        task.add_done_callback(lambda f: (LEVEL_TASKS.pop(key, None), f.exception()))
    return Response(status_code=202, headers=hdr)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "yt_dlp": yt_dlp.version.__version__}


@app.get("/search")
async def search(q: str, kind: str = "music", episode: str = ""):
    q = q.strip()[:200]
    if not q:
        raise HTTPException(400, "Say what to play.")
    try:
        if kind == "radio":
            items = await radio_search(q)
        elif kind == "podcast":
            items = await podcast_search(q, episode.strip()[:200])
        else:
            items = await asyncio.to_thread(yt_search, q)
            if items:                            # download the first song while the avatar talks
                first = TOKENS[items[0]["url"].split("/")[-1]]["src"]
                asyncio.ensure_future(yt_file(first)).add_done_callback(lambda f: f.exception())
    except httpx.HTTPError as e:
        raise HTTPException(502, f"The {kind} search didn't answer: {e}")
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(502, "YouTube search failed: " + str(e).replace("ERROR: ", ""))
    return {"items": items}


@app.post("/prefetch/{tok}")
async def prefetch(tok: str):
    """Start downloading a song before it's needed, so the next one starts without a wait."""
    t = TOKENS.get(tok)
    if t and t["kind"] == "music" and not (CACHE / f"{t['src']}.m4a").exists():
        task = asyncio.ensure_future(yt_file(t["src"]))
        task.add_done_callback(lambda f: f.exception())     # failures show up when it's played
    return Response(status_code=204)


PASS = ("content-type", "content-length", "content-range", "accept-ranges")


@app.get("/stream/{tok}")
async def stream(tok: str, request: Request):
    t = TOKENS.get(tok)
    if not t or t["exp"] < time.time():
        raise HTTPException(404, "That search has expired. Ask for it again.")
    if t["kind"] == "music":
        try:
            path = await yt_file(t["src"])
        except Exception as e:
            raise HTTPException(502, "YouTube wouldn't play that: " + str(e).replace("ERROR: ", "")[:300])
        # FileResponse answers Range requests, which iPhones insist on.
        return FileResponse(path, media_type="audio/mp4", headers={"cache-control": "no-store"})

    headers = {}
    url = t["src"]
    if request.headers.get("range") and t["kind"] == "podcast":
        headers["Range"] = request.headers["range"]

    req = http.build_request("GET", url, headers=headers)
    try:
        up = await http.send(req, stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Couldn't reach the stream: {e}")
    if up.status_code >= 400:
        await up.aclose()
        raise HTTPException(502, f"The stream answered {up.status_code}.")

    out = {k: up.headers[k] for k in PASS if k in up.headers}
    if t["kind"] == "radio":
        out.pop("content-length", None)
        if not out.get("content-type", "").startswith("audio"):
            out["content-type"] = "audio/mpeg"
    out["cache-control"] = "no-store"

    async def body():
        try:
            async for chunk in up.aiter_raw():
                yield chunk
        finally:
            await up.aclose()

    return StreamingResponse(body(), status_code=up.status_code, headers=out)
