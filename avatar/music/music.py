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


def item(kind, src, title, subtitle="", art="", duration=None, live=False):
    return {"kind": kind, "title": title or "Untitled", "subtitle": subtitle or "",
            "art": art or "", "duration": duration, "live": live,
            "url": "stream/" + remember(kind, src)}


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
    return item("music", e["id"], e.get("title"),
                e.get("channel") or e.get("uploader") or "",
                yt_thumb(e), e.get("duration"))


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
    prune()
    return done


def prune():
    files = sorted((f for f in CACHE.glob("*.m4a")), key=lambda f: f.stat().st_mtime, reverse=True)
    total = 0
    for f in files:
        total += f.stat().st_size
        if total > CACHE_BYTES:
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
                        s.get("favicon") or "", None, True))
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
                    seconds(it.findtext(ITUNES + "duration"))))
    if episode:                                  # "the one about ..." / an episode number
        words = [w for w in re.findall(r"\w+", episode.lower()) if len(w) > 1]
        scored = [(sum(w in e[0].lower() for w in words), -i, e) for i, e in enumerate(eps)]
        best = max(scored, default=None)
        if best and best[0] > 0:
            i = eps.index(best[2])
            eps = eps[i:]
    name = show.get("collectionName") or ""
    return [item("podcast", url, title, " · ".join(x for x in (name, when) if x),
                 img or show_art, dur)
            for title, url, when, img, dur in eps[:15]]


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
