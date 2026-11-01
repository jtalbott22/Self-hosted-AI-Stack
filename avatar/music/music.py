"""Music for the avatar page: search and stream songs, radio and podcasts.

    GET /health
    GET /search?q=...&kind=music|radio|podcast   -> {"items": [...], "more": ...}
    GET /more/{token}                            -> the songs that follow a song
    GET /stream/{token}                          -> the audio, same origin
    POST /prefetch/{token}[?warm=1]              -> get the next song ready
    GET /levels/{token}                          -> the visualizer's frames
    GET /lyrics/{token}                          -> a song's lyrics, timed when possible
                                                    (and matched to the singing, with STT_URL)

Sources, none of which need an account or a key:
    music    YouTube, through yt-dlp. The first hit is followed by its YouTube
             "mix", so a song keeps going with related songs. Song-length
             hits win over hour-long mixes unless that's what was asked for.
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
import json
import os
import re
import secrets
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from pathlib import Path

import httpx
import numpy as np
import yt_dlp
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

UA = "avatar-music/1.0 (self-hosted voice assistant)"
RADIO_API = os.environ.get("RADIO_API", "https://all.api.radio-browser.info")
TOKEN_TTL = 12 * 3600            # how long a search result stays playable
CACHE = Path(os.environ.get("CACHE_DIR", "/cache"))
CACHE_BYTES = int(float(os.environ.get("CACHE_GB", "2")) * 1024**3)
MIX_LEN = 25
SONG = (60, 600)                 # what counts as song length, in seconds

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
# YouTube makes most of its players wait out the length of an ad (5 to 15 s)
# before the audio may be fetched. Its visionOS player doesn't, so a song is
# here in about two seconds; yt-dlp's usual players are the fallback.
FAST = {"extractor_args": {"youtube": {"player_client": ["visionos"]}}}
# Words that mean a long video is what's wanted, not a song.
LONG_OK = re.compile(r"\b(mix(es)?|hours?|hrs?|album|full|concert|set|playlist|compilation|medley|"
                     r"ambient|sleep|study|nonstop|non-stop|marathon|soundtrack|ost|asmr)\b", re.I)

def yt_thumb(e):
    vid = e.get("id")
    # 16:9 with no letterbox bars; the page falls back to hqdefault for the
    # few (old, low-resolution) videos that don't have it
    return f"https://i.ytimg.com/vi/{vid}/hq720.jpg" if vid else ""


def yt_entry(e):
    who = e.get("channel") or e.get("uploader") or ""
    it = item("music", e["id"], e.get("title"), who, yt_thumb(e), e.get("duration"),
              link=f"https://www.youtube.com/watch?v={e['id']}", link_label="Watch on YouTube",
              credit=f"Uploaded to YouTube by {who}" if who else "From YouTube",
              credit_link=f"https://www.youtube.com/channel/{e['channel_id']}" if e.get("channel_id") else "")
    tok = it["url"].split("/")[-1]
    TOKENS[tok]["meta"] = (e.get("title") or "", who, e.get("duration"))
    it["lyrics"] = "lyrics/" + tok
    return it


def yt_list(url: str, n: int = 10):
    with yt_dlp.YoutubeDL({**YDL_FLAT, "playlistend": n}) as y:
        res = y.extract_info(url, download=False) or {}
    # live streams have no length, and can't be downloaded
    return [e for e in res.get("entries") or []
            if e and e.get("id") and e.get("duration") and e["duration"] < 3 * 3600
            and e.get("live_status") not in ("is_live", "is_upcoming")
            and "/shorts/" not in (e.get("url") or "")]


def song_length(e):
    return SONG[0] <= (e.get("duration") or 0) <= SONG[1]


def yt_search(q: str):
    """The first hit of song length, so it starts quickly and doesn't play for
    hours: "lofi" finds a lofi song, not a 12-hour stream. Unless the words ask
    for something long ("1 hour", "full album", "live set"), then the top hit.
    Returns (first, other hits)."""
    hits = yt_list(f"ytsearch10:{q}")
    if LONG_OK.search(q):
        return (hits[0], hits[1:]) if hits else (None, [])
    songs = [e for e in hits if song_length(e)]
    if not songs:
        # genres and moods mostly turn up long mixes; ask again for 4-20 minute videos
        try:
            more = yt_list("https://www.youtube.com/results?search_query="
                           + quote_plus(q) + "&sp=EgIYAw%3D%3D")
        except Exception:
            more = []
        songs = [e for e in more if song_length(e)] or sorted(
            (e for e in more + hits if e["duration"] <= 1200), key=lambda e: e["duration"])
        hits = hits + more
    first = songs[0] if songs else (hits[0] if hits else None)
    return first, [e for e in hits if e is not first and song_length(e)]


def yt_mix(first: dict, fallback: list):
    """The songs that follow a song: its YouTube mix, related songs the way
    YouTube Music keeps playing. One request for the list; no media."""
    try:
        with yt_dlp.YoutubeDL(YDL_MIX) as y:
            mix = y.extract_info(
                f"https://www.youtube.com/watch?v={first['id']}&list=RD{first['id']}",
                download=False) or {}
        rest = [e for e in mix.get("entries") or [] if e and e.get("id") and e["id"] != first["id"]
                and song_length(e)]
    except Exception:
        rest = []
    if len(rest) < 3:
        rest = fallback
    return [yt_entry(e) for e in rest[:MIX_LEN]]


# Looking a song up (which player, which file) without downloading it, so a
# skip only has the download left. Media URLs last about six hours.
INFO: dict[str, tuple[float, dict]] = {}
INFO_TTL = 3 * 3600


def yt_info(vid: str) -> dict:
    hit = INFO.get(vid)
    if hit and time.time() - hit[0] < INFO_TTL:
        return hit[1]
    url = f"https://www.youtube.com/watch?v={vid}"
    info = None
    for extra in (FAST, {}):
        try:
            with yt_dlp.YoutubeDL({**YDL_AUDIO, **extra, "skip_download": True}) as y:
                info = y.extract_info(url, download=False)
            break
        except yt_dlp.utils.DownloadError:
            if not extra:
                raise
    for k in [k for k, v in INFO.items() if time.time() - v[0] > INFO_TTL]:
        INFO.pop(k, None)
    INFO[vid] = (time.time(), info)
    return info


def yt_fetch(vid: str) -> Path:
    """Download one song into the cache, once. A song is a few MB and takes
    about two seconds all told; the page asks for the next one ahead of time."""
    done = CACHE / f"{vid}.m4a"
    if done.exists():
        os.utime(done)
        return done
    work = CACHE / f"tmp-{vid}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        try:
            with yt_dlp.YoutubeDL({**YDL_AUDIO, "paths": {"home": str(work)}}) as y:
                y.process_ie_result(yt_info(vid), download=True)
        except yt_dlp.utils.DownloadError:
            # that player's file was refused: start over with yt-dlp's usual players
            INFO.pop(vid, None)
            for f in work.iterdir():
                f.unlink(missing_ok=True)
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
    INFO.pop(vid, None)
    prune()
    return done                                 # the visualizer's levels are worked out when asked for


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


# At most two songs download at once: the one playing and the one after it.
DOWNLOADS = asyncio.Semaphore(2)


async def _fetch_one(vid: str) -> Path:
    async with DOWNLOADS:
        return await asyncio.to_thread(yt_fetch, vid)


WARMING: dict[str, asyncio.Task] = {}


def warm(vid: str):
    """Look the next song up now, download it later (or on a skip)."""
    if vid in INFO or vid in WARMING or vid in FETCHING or (CACHE / f"{vid}.m4a").exists():
        return
    task = asyncio.create_task(asyncio.to_thread(yt_info, vid))
    WARMING[vid] = task
    task.add_done_callback(lambda f: (WARMING.pop(vid, None), f.exception()))


async def yt_file(vid: str) -> Path:
    """One download per song, however many requests ask for it at once."""
    task = FETCHING.get(vid)
    if task is None:
        task = asyncio.create_task(_fetch_one(vid))
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
# Lyrics, from LRCLIB (lrclib.net): a free, open lyrics database, most of it
# timed line by line. YouTube titles are "Artist - Song (Official Video)";
# the artist and song are worked out from that, and of the matches the one
# closest in length to the video wins, so the timing lines up.
# --------------------------------------------------------------------------
LRCLIB = "https://lrclib.net/api"
LYRICS = CACHE / "lyrics"
LYRICS.mkdir(parents=True, exist_ok=True)
_NOISE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")         # (Official Video) [4K] (feat. X) ...
_FEAT = re.compile(r"\s+(ft\.?|feat\.?|featuring|with)\s+.*$", re.I)
_TAIL = re.compile(r"\s*(\||//).*$|\s+(official\s+)?(music\s+)?(video|audio|lyrics?|visuali[sz]er|hd|4k)\s*$", re.I)


def guess_song(title: str, channel: str):
    t = _TAIL.sub("", _NOISE.sub("", title)).strip(" -–—\"'")
    parts = re.split(r"\s+[-–—]\s+", t, maxsplit=1)
    if len(parts) == 2:
        artist, track = parts
    else:
        artist = re.sub(r"\s*-\s*Topic$|VEVO$|\s+Official$", "", channel or "", flags=re.I)
        track = t
    return _FEAT.sub("", artist).strip(), _FEAT.sub("", track).strip(" \"'")


def parse_lrc(text: str):
    lines = []
    for raw in (text or "").splitlines():
        stamps = re.findall(r"\[(\d+):(\d+(?:\.\d+)?)\]", raw)
        words = re.sub(r"\[[^\]]*\]", "", raw).strip()
        for m, sec in stamps:
            lines.append([round(int(m) * 60 + float(sec), 2), words])
    return sorted(lines)


async def find_lyrics(title: str, channel: str, duration):
    artist, track = guess_song(title, channel)
    found = []
    for params in ({"track_name": track, "artist_name": artist}, {"q": f"{artist} {track}"}):
        if not track:
            break
        r = await http.get(f"{LRCLIB}/search", params=params)
        r.raise_for_status()                     # busy (503): don't remember it as "no lyrics"
        found = [x for x in r.json() if not x.get("instrumental") and (x.get("syncedLyrics") or x.get("plainLyrics"))]
        if found:
            break
    if not found:
        return {"lines": None, "plain": None}
    d = duration or 0
    timed = [x for x in found if x.get("syncedLyrics")]
    best = min(timed or found, key=lambda x: abs((x.get("duration") or 0) - d) if d else 0)
    return {"lines": parse_lrc(best.get("syncedLyrics") or "") or None,
            "plain": (best.get("plainLyrics") or "").strip() or None,
            "track": best.get("trackName"), "artist": best.get("artistName"),
            "lyricsDuration": best.get("duration"), "duration": duration,
            "credit": "Lyrics from LRCLIB", "creditLink": "https://lrclib.net/"}


# --------------------------------------------------------------------------
# Matching the lyrics to the singing. LRCLIB's timing is for the record, so
# a video with a skit before the song (or a pause in the middle) runs ahead
# of it. Whisper (the speech-to-text service) listens to the song in 30 s
# pieces; each lyric line is looked for in what it heard, and the lines are
# slipped by however far they turn up from where LRCLIB put them. A pause in
# the middle shows up as a second, larger slip from that line on.
# Set STT_URL to an OpenAI-style speech-to-text service to turn it on.
# --------------------------------------------------------------------------
STT_URL = os.environ.get("STT_URL", "").rstrip("/")
STT_MODEL = os.environ.get("STT_MODEL", "Systran/faster-whisper-small.en")
PIECE = 30                        # seconds per piece; longer ones send Whisper into loops
JITTER = 1.2                      # how far Whisper's word times wander
SLIP_COST = 1.5                   # a change of slip needs about two lines behind it
EARS = asyncio.Lock()             # one piece at a time: the speech service also hears the user
HEARING: dict[str, asyncio.Task] = {}
DEAF: dict[str, float] = {}       # songs whose listen failed, and when


def words_of(s: str):
    out = []
    for w in re.sub(r"[’‘`]", "'", s.lower()).split():
        w = re.sub(r"[^a-z0-9]", "", w)
        if w.endswith("ing"):
            w = w[:-1]                                   # lovin' = loving
        if w:
            out.append(w)
    return out


def spots(toks, heard, keep=8):
    """Where a lyric line might be in what was heard: [(time, score)]."""
    n, seq, found = len(toks), [w for w, _ in heard], []
    for k, w in enumerate(seq):
        if w not in toks[:3]:
            continue                                     # a line is found by how it starts
        sm = SequenceMatcher(None, toks, seq[k:k + n + 2], autojunk=False)
        score = sum(b.size for b in sm.get_matching_blocks()) / n
        if score >= 0.5:
            found.append((heard[k][1] - 0.3 * toks.index(w), score))
    found.sort(key=lambda c: -c[1])
    picked = []
    for c in found:
        if all(abs(c[0] - p[0]) > 1.5 for p in picked):
            picked.append(c)
            if len(picked) >= keep:
                break
    return picked


def match_lines(lines, heard):
    """The slip for every line, or None when too little was heard to say.
    Finds the best run of lines in order whose times keep rising, where
    changing the slip costs something (so one stray match can't move it)."""
    toks = [words_of(t) for _, t in lines]
    cands = []
    for (t, _), tk in zip(lines, toks):
        cands.append([] if len(tk) < 2 else
                     [(tm, s * min(1.0, len(tk) / 4), tm - t) for tm, s in spots(tk, heard)])
    best = {}
    for i, cs in enumerate(cands):
        for c, (tm, s, off) in enumerate(cs):
            top, prev = s, None
            for j in range(max(0, i - 20), i):
                for c2, (tm2, _, off2) in enumerate(cands[j]):
                    if tm2 < tm - 0.2:
                        v = best[(j, c2)][0] + s - (0 if abs(off - off2) < JITTER else SLIP_COST)
                        if v > top:
                            top, prev = v, (j, c2)
            best[(i, c)] = (top, prev)
    if not best:
        return None
    node, path = max(best, key=lambda k: best[k][0]), []
    while node:
        path.append((node[0], cands[node[0]][node[1]][2]))
        node = best[node][1]
    path.reverse()
    if len(path) < max(4, 0.15 * len(lines)):
        return None
    runs = []                                            # stretches of lines that agree
    for i, off in path:
        if runs and abs(off - runs[-1]["at"]) < JITTER:
            runs[-1]["offs"].append(off)
        else:
            runs.append({"first": i, "offs": [off]})
        runs[-1]["at"] = sorted(runs[-1]["offs"])[len(runs[-1]["offs"]) // 2]
    runs = [r for r in runs if len(r["offs"]) >= 2] or runs
    slips, r = [], 0
    for i in range(len(lines)):
        while r + 1 < len(runs) and runs[r + 1]["first"] <= i:
            r += 1
        at = runs[r]["at"]
        slips.append(0.0 if abs(at) < 0.5 else round(at, 2))   # under half a second: LRCLIB's own is better
    return {"slips": slips, "matched": len(path), "runs": len(runs)}


def piece(path: Path, start: int) -> bytes:
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(start), "-t", str(PIECE), "-i", str(path),
                           "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
                          capture_output=True, check=True, timeout=60).stdout


async def listen(vid: str, out: Path):
    """Hear the song and save the slips into its lyrics file as they firm up."""
    res = json.loads(out.read_text())
    lines, length = res["lines"], res.get("duration") or 0
    path = await yt_file(vid)
    if not length:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True)
        length = float(p.stdout.strip() or 0)
    heard = []
    for start in range(0, int(length) + 1, PIECE):
        wav = await asyncio.to_thread(piece, path, start)
        if len(wav) < 16000:                             # under half a second left
            break
        async with EARS:
            r = await http.post(f"{STT_URL}/audio/transcriptions", timeout=120,
                                files={"file": ("piece.wav", wav, "audio/wav")},
                                data={"model": STT_MODEL, "response_format": "verbose_json",
                                      "timestamp_granularities[]": ["word"], "vad_filter": "false"})
        r.raise_for_status()
        for w in r.json().get("words") or []:
            heard += [(t, start + w["start"]) for t in words_of(w["word"])]
        m = match_lines(lines, heard)
        last = start + PIECE >= length
        # Already on time, and the video is the record's length, so no pause hides further on.
        on_time = (m and m["matched"] >= 6 and m["runs"] == 1 and not any(m["slips"])
                   and abs(length - (res.get("lyricsDuration") or 0)) < 3)
        res["heard"] = {**(m or {}), "done": bool(last or on_time), "upto": start + PIECE}
        out.write_text(json.dumps(res))
        if last or on_time:
            break
    print("lyrics:", vid, "heard", res["heard"].get("matched", 0), "of", len(lines), "lines, slip",
          sorted(set(res["heard"].get("slips") or [0])))


def start_listening(vid: str, out: Path):
    if not STT_URL or vid in HEARING or time.time() - DEAF.get(vid, 0) < 600:
        return
    task = asyncio.create_task(listen(vid, out))
    HEARING[vid] = task

    def done(t):
        HEARING.pop(vid, None)
        if t.exception():
            DEAF[vid] = time.time()
            print("lyrics:", vid, "couldn't listen:", repr(t.exception()))
    task.add_done_callback(done)


def as_heard(res: dict, vid: str) -> dict:
    """The lyrics with the slips applied, and how the listening is going."""
    h = res.get("heard") or {}
    out = {k: v for k, v in res.items() if k != "heard"}
    if res.get("lines") and h.get("slips"):
        out["lines"] = sorted([round(t + s, 2), txt] for (t, txt), s in zip(res["lines"], h["slips"]))
    out["matched"] = ("listening" if vid in HEARING else
                      "" if not h.get("done") else
                      "moved" if any(h.get("slips") or []) else
                      "on time" if h.get("slips") else "")
    return out


@app.get("/lyrics/{tok}")
async def lyrics(tok: str):
    t = TOKENS.get(tok)
    if not t or "meta" not in t:
        raise HTTPException(404, "No lyrics for that.")
    vid, out = t["src"], LYRICS / f"{t['src']}.json"
    if out.exists() and (time.time() - out.stat().st_mtime < 7 * 86400 or out.stat().st_size > 60):
        res = json.loads(out.read_text())
    else:
        try:
            res = await find_lyrics(*t["meta"])
        except (httpx.HTTPError, ValueError):
            raise HTTPException(502, "The lyrics service didn't answer.")
        out.write_text(json.dumps(res))             # misses are kept a week, then asked again
    if res.get("lines") and not (res.get("heard") or {}).get("done"):
        start_listening(vid, out)
    return as_heard(res, vid)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
@app.post("/log")
async def page_log(request: Request):
    """Playback reports from the page (refused plays, stalls, errors), so a
    phone's problems show up in `docker logs music`."""
    body = (await request.body())[:2000].decode("utf-8", "replace").replace("\n", " ")
    print("page:", body, flush=True)
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"ok": True, "yt_dlp": yt_dlp.version.__version__}


@app.get("/search")
async def search(q: str = "", kind: str = "music", episode: str = "",
                 id: str = "", title: str = "", by: str = ""):
    """id (kind music): play that YouTube video again, as from the history,
    with title/by from the page, so it costs one request instead of two."""
    q = q.strip()[:200]
    if id and not re.fullmatch(r"[A-Za-z0-9_-]{11}", id):
        raise HTTPException(400, "That isn't a YouTube video id.")
    if not q and not (id and kind == "music"):
        raise HTTPException(400, "Say what to play.")
    try:
        if kind == "music":
            if id:
                first, rest = {"id": id, "title": title[:300] or "YouTube video", "channel": by[:200]}, []
            else:
                first, rest = await asyncio.to_thread(yt_search, q)
            if not first:
                return {"items": []}
            # The song downloads while the avatar answers. The songs after it
            # are a separate request (/more), so the answer doesn't wait for them.
            asyncio.ensure_future(yt_file(first["id"])).add_done_callback(lambda f: f.exception())
            it = yt_entry(first)
            tok = it["url"].split("/")[-1]
            TOKENS[tok]["follow"] = (first, rest[:MIX_LEN])
            return {"items": [it], "more": "more/" + tok}
        elif kind == "radio":
            items = await radio_search(q)
        else:
            items = await podcast_search(q, episode.strip()[:200])
    except httpx.HTTPError as e:
        raise HTTPException(502, f"The {kind} search didn't answer: {e}")
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(502, "YouTube search failed: " + str(e).replace("ERROR: ", ""))
    return {"items": items}


@app.get("/more/{tok}")
async def more(tok: str):
    """The songs to follow a song from /search: its YouTube mix."""
    t = TOKENS.get(tok)
    if not t or "follow" not in t:
        raise HTTPException(404, "That search has expired. Ask for it again.")
    first, rest = t["follow"]
    try:
        items = await asyncio.to_thread(yt_mix, first, rest)
    except Exception:
        items = []
    return {"items": items}


@app.post("/prefetch/{tok}")
async def prefetch(tok: str, warm_only: int = Query(0, alias="warm")):
    """Get a song ready before it's needed. warm=1 only looks it up (a few
    small requests, no media), so a skip has just the download left; without
    it, the song downloads too, which the page asks for in the last minute."""
    t = TOKENS.get(tok)
    if t and t["kind"] == "music" and not (CACHE / f"{t['src']}.m4a").exists():
        if warm_only:
            warm(t["src"])
        else:
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
