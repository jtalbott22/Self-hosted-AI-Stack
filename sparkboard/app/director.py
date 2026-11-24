"""
Model director: one front door for the two llama.cpp servers, and the thing
that decides which of them is loaded, with how much context, and how fast.

Measured on this box (RTX 4070 12 GB, 54 GB WSL RAM), 2026-11-23. VRAM figures
are measured alone; anything that reads ~11.7 GB or more has overflowed into
Windows' shared GPU memory, which still "works" but at unpredictable speed
(it is what made the first version of this hang), so every layout below
leaves at least 1.2 GB free.

  Qwen 131k, experts on CPU ........ 5.9 GB   28 tok/s
  Qwen 131k, 10 expert layers GPU .. 10.6 GB  41 tok/s   (alone)
  Qwen 262k, 6 expert layers GPU ... 10.9 GB  37 tok/s   (alone)
  Inkling 32k, experts on CPU, ub 256 ...... 5.0 GB   12 tok/s
  Inkling 32k, experts 6-15 on GPU ......... 10.4 GB  26 tok/s   (alone)
  Inkling 64k, experts 6-12 on GPU ......... 10.9 GB  20.5 tok/s (alone)
  Inkling 128k, experts 6-7 on GPU ......... 10.6 GB  13.4 tok/s (alone)

Inkling has 24 physical layers and runs layers 6-17 three times per token,
so the experts of those looped layers are the ones worth putting on the GPU
(--n-cpu-moe offloads from the end and barely helps). -ub 256 instead of 512
saves ~1.6 GB of compute buffer, which is what makes 32k fit beside Qwen.

An idle model costs the busy one nothing, so a model is only stopped when
its VRAM buys the other one speed or context. Either model "ramps": a
session of requests (BOOST_AFTER inside BOOST_WINDOW, the other model quiet)
earns its solo fast layout, and a prompt too big for the current context
moves it to a bigger one. Idle drops everything back to both loaded.

Every Qwen variant carries the network alias `llama` and binds host :8000,
so Home Assistant, the avatar and Open WebUI keep working in every layout
that has Qwen in it. Inkling variants share `llama-inkling` / :8001. The
Inkling-only layouts stop Qwen, so Home Assistant has no model while they
are up (DIRECTOR_ALLOW_QWEN_STOP=0 forbids them).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import re
import time

import httpx

from . import config

log = logging.getLogger("sparkboard.director")


def _env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _num(name, default):
    try:
        return float(_env(name, default))
    except (TypeError, ValueError):
        return float(default)


ENABLED = _env("DIRECTOR", "0") == "1"
API_KEY = _env("DIRECTOR_LLAMA_API_KEY", _env("LLAMA_API_KEY", ""))
STATE_PATH = _env("DIRECTOR_STATE", "/data/director.json")

# A "session" (this many requests inside the window, the other model quiet
# for GRACE seconds) earns that model's solo fast layout. The first request is always served at once in
# whatever layout is up, so one question never waits on a restart.
BOOST_AFTER = int(_num("DIRECTOR_BOOST_AFTER", 2))
BOOST_WINDOW = _num("DIRECTOR_BOOST_WINDOW", 180)
GRACE = _num("DIRECTOR_GRACE", _num("DIRECTOR_INKLING_GRACE", 300))
# Whether Inkling may take the whole GPU (Qwen stopped, so Home Assistant has
# no model) for speed or for prompts over 32k.
ALLOW_QWEN_STOP = _env("DIRECTOR_ALLOW_QWEN_STOP", "1") == "1"
# Local time window held in the shared layout, so the 02:00 nightly
# `docker compose up -d` never collides with a variant holding a port.
QUIET = _env("DIRECTOR_QUIET", "01:45-02:45")
COMFY_URL = _env("DIRECTOR_COMFY_URL", "http://comfyui:8188")
HEALTH_TIMEOUT = _num("DIRECTOR_HEALTH_TIMEOUT", 300)

UPSTREAM = {"qwen": _env("DIRECTOR_QWEN_URL", "http://llama:8000"),
            "inkling": _env("DIRECTOR_INKLING_URL", "http://llama-inkling:8000")}
MODELS = tuple(UPSTREAM)

# Container -> (model, context). Every variant is a compose service.
CONTAINERS = {
    "llama": ("qwen", 131072),
    "llama-boost": ("qwen", 131072),
    "llama-long": ("qwen", 262144),
    "llama-inkling": ("inkling", 32768),
    "llama-inkling-fast": ("inkling", 32768),
    "llama-inkling-64k": ("inkling", 65536),
    "llama-inkling-128k": ("inkling", 131072),
}
MANAGED = set(CONTAINERS)

# owner: the model whose idleness ends the layout. restore: seconds idle.
# solo: takes the whole GPU (yields to ComfyUI).
LAYOUTS = {
    "shared": {"containers": ["llama", "llama-inkling"], "owner": None, "restore": 0,
               "label": "Both loaded", "detail": "Qwen 131k at 28 tok/s, Inkling 32k at 12 tok/s"},
    "boost": {"containers": ["llama-boost"], "owner": "qwen", "restore": 600,
              "label": "Qwen boost", "detail": "Qwen 131k at 41 tok/s; Inkling parked"},
    "long": {"containers": ["llama-long"], "owner": "qwen", "restore": 600,
             "label": "Qwen long", "detail": "Qwen 262k at 37 tok/s; Inkling parked"},
    "inkboost": {"containers": ["llama-inkling-fast"], "owner": "inkling", "restore": 300,
                 "label": "Inkling boost",
                 "detail": "Inkling 32k at 26 tok/s; Qwen stopped (Home Assistant has no model)"},
    "ink64": {"containers": ["llama-inkling-64k"], "owner": "inkling", "restore": 300,
              "label": "Inkling 64k",
              "detail": "Inkling 64k at 20 tok/s; Qwen stopped (Home Assistant has no model)"},
    "ink128": {"containers": ["llama-inkling-128k"], "owner": "inkling", "restore": 300,
               "label": "Inkling 128k",
               "detail": "Inkling 128k at 13 tok/s; Qwen stopped (Home Assistant has no model)"},
}
for _k, _l in LAYOUTS.items():
    _l["solo"] = _k != "shared"
BOOST_LAYOUT = {"qwen": "boost", "inkling": "inkboost"}
MODES = ("auto", "shared", "boost", "inkboost")
for _l in LAYOUTS.values():
    _l["ctx"] = {m: 0 for m in MODELS}
    for _c in _l["containers"]:
        _m, _n = CONTAINERS[_c]
        _l["ctx"][_m] = _n

_DATA_URL = re.compile(r'data:[^"]{200,}')


def _need_tokens(raw: bytes) -> int:
    """Context a request needs: a generous prompt estimate plus some output."""
    text = _DATA_URL.sub("", raw.decode("utf-8", "replace"))
    prompt = int(len(text) / 3.2)
    out = 1024
    try:
        body = json.loads(raw)
        mt = body.get("max_tokens") or body.get("max_completion_tokens")
        if isinstance(mt, int) and mt > 0:
            out = mt
    except (ValueError, AttributeError):
        pass
    # llama.cpp only fails when the prompt itself won't fit; a long answer is
    # just cut off at the context. So reserve a bounded slice for output.
    return prompt + min(out, 4096)


def _in_quiet(now=None) -> bool:
    try:
        a, b = QUIET.split("-")
        ah, am = map(int, a.split(":"))
        bh, bm = map(int, b.split(":"))
    except ValueError:
        return False
    t = time.localtime(now or time.time())
    m = t.tm_hour * 60 + t.tm_min
    lo, hi = ah * 60 + am, bh * 60 + bm
    return lo <= m < hi if lo <= hi else (m >= lo or m < hi)


class Director:
    def __init__(self):
        self.docker = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=config.DOCKER_SOCK),
            base_url="http://docker", timeout=httpx.Timeout(60.0, connect=5.0))
        self.http = httpx.AsyncClient(timeout=None)
        self.probe = httpx.AsyncClient(timeout=5.0)
        self.layout = "shared"
        self.switching: str | None = None
        self._lock = asyncio.Lock()
        self._settled = asyncio.Event()
        self._settled.set()
        self.inflight = {m: 0 for m in MODELS}
        self.last_req = {m: 0.0 for m in MODELS}
        self.recent = {m: collections.deque(maxlen=50) for m in MODELS}
        self.events = collections.deque(maxlen=60)
        self.block_until = 0.0
        self.mode = "auto"              # see MODES
        self.parked: set[str] = set()   # models the user stopped by hand
        self.stats = {"forwarded": 0, "switches": 0, "held_ms_total": 0}
        self._task = None
        self._load()

    # ------------------------------------------------------------ persistence

    def _load(self):
        try:
            with open(STATE_PATH) as f:
                d = json.load(f)
            if d.get("mode") in MODES:
                self.mode = d["mode"]
            self.parked = set(d.get("parked", [])) & set(MODELS)
        except (OSError, ValueError):
            pass

    def _save(self):
        try:
            tmp = STATE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"mode": self.mode, "parked": sorted(self.parked)}, f)
            os.replace(tmp, STATE_PATH)
        except OSError as exc:
            log.warning("director state not saved: %s", exc)

    def event(self, text):
        log.info("director: %s", text)
        self.events.append({"ts": time.time(), "text": text})

    # ----------------------------------------------------------------- docker

    async def cstate(self, name) -> str:
        try:
            r = await self.docker.get(f"/containers/{name}/json")
            if r.status_code == 404:
                return "missing"
            return r.json().get("State", {}).get("Status", "unknown")
        except Exception:
            return "unknown"

    async def cact(self, name, action):
        q = "?t=15" if action in ("stop", "restart") else ""
        r = await self.docker.post(f"/containers/{name}/{action}{q}")
        if r.status_code not in (204, 304):
            raise RuntimeError(f"{action} {name}: {r.status_code} {r.text[:200]}")

    async def running(self) -> set[str]:
        out = set()
        for c in CONTAINERS:
            if await self.cstate(c) == "running":
                out.add(c)
        return out

    def _reset_clients(self):
        # Variants share DNS names, so pooled connections may point at a
        # container that was just stopped. Start clean after every switch.
        old = (self.http, self.probe)
        self.http = httpx.AsyncClient(timeout=None)
        self.probe = httpx.AsyncClient(timeout=5.0)
        for c in old:
            asyncio.create_task(c.aclose())

    def _auth(self):
        return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

    async def _healthy(self, model, timeout=HEALTH_TIMEOUT) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            try:
                if (await self.probe.get(UPSTREAM[model] + "/health")).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
        return False

    async def _busy_direct(self, model) -> bool:
        """True if the model is generating for a client that bypassed us."""
        try:
            r = await self.probe.get(UPSTREAM[model] + "/slots", headers=self._auth())
            return any(s.get("is_processing") for s in r.json())
        except Exception:
            return False

    async def _comfy_busy(self) -> bool:
        try:
            q = (await self.probe.get(COMFY_URL + "/queue")).json()
            return bool(q.get("queue_running") or q.get("queue_pending"))
        except Exception:
            return False

    async def _warm(self, model):
        # One token, so the first real reply doesn't pay for paging weights in.
        body = {"model": model, "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}]}
        if model == "inkling":
            body["chat_template_kwargs"] = {"reasoning_effort": 0}
        try:
            await self.http.post(UPSTREAM[model] + "/v1/chat/completions",
                                 json=body, headers=self._auth(), timeout=180)
        except Exception:
            pass

    # ---------------------------------------------------------------- layouts

    def _targets(self, layout):
        return [c for c in LAYOUTS[layout]["containers"]
                if CONTAINERS[c][0] not in self.parked]

    async def _rest(self, now) -> str:
        """Where to sit when idle: both loaded, or the survivor's fast layout
        when the other model was stopped by hand."""
        live = [m for m in MODELS if m not in self.parked]
        if len(live) == 1:
            r = BOOST_LAYOUT[live[0]]
            if self._allowed(r, now) and not await self._comfy_busy():
                return r
        return "shared"

    async def _drain(self, models, limit=900):
        end = time.time() + limit
        while time.time() < end:
            if not any(self.inflight[m] for m in models):
                if not [m for m in models if await self._busy_direct(m)]:
                    return True
            await asyncio.sleep(0.5)
        return False

    async def _apply(self, target, why):
        self.switching = target
        t0 = time.time()
        want = set(self._targets(target))
        up = await self.running()
        losing = {CONTAINERS[c][0] for c in up - want}
        await self._drain(sorted(losing))
        for c in sorted(up - want):
            await self.cact(c, "stop")
        for c in sorted(want - up):
            await self.cact(c, "start")
        self._reset_clients()
        models = sorted({CONTAINERS[c][0] for c in want})
        ok = await asyncio.gather(*(self._healthy(m) for m in models))
        if not all(ok):
            raise RuntimeError(f"{target}: not healthy after {HEALTH_TIMEOUT:.0f}s")
        self.layout = target
        self.stats["switches"] += 1
        self.event(f"{LAYOUTS[target]['label']} ({why}) in {time.time() - t0:.0f}s")
        await asyncio.gather(*(self._warm(m) for m in models))

    async def switch(self, target, why):
        # Run as its own task and shield it: a client that disconnects while
        # waiting must not cancel a layout change halfway through.
        await asyncio.shield(asyncio.ensure_future(self._switch(target, why)))

    async def _switch(self, target, why):
        async with self._lock:
            if target == self.layout and not await self._drifted():
                return
            self._settled.clear()
            try:
                await self._apply(target, why)
            except Exception as exc:
                self.event(f"switch to {target} failed: {exc}; back to both, "
                           "no switching for 30 min")
                self.block_until = time.time() + 1800
                try:
                    await self._apply("shared", "recovery")
                except Exception as exc2:
                    self.event(f"recovery failed: {exc2}")
            finally:
                self.switching = None
                self._settled.set()

    async def _drifted(self) -> bool:
        return await self.running() != set(self._targets(self.layout))

    # ----------------------------------------------------------------- policy

    def _allowed(self, layout, now) -> bool:
        if layout == "shared":
            return True
        if _in_quiet(now) or now < self.block_until:
            return False
        if not self._targets(layout):
            return False    # e.g. an Inkling-only layout while Inkling is parked
        if self.mode == "shared" and not self.parked:
            return False    # "Both" means both stay loaded
        if LAYOUTS[layout]["ctx"]["qwen"] == 0 and not ALLOW_QWEN_STOP \
                and "qwen" not in self.parked:
            return False
        return True

    def _pick(self, model, need, now) -> str:
        """Layout to serve `model` with `need` tokens of context."""
        cur = LAYOUTS[self.layout]
        if cur["ctx"][model] >= need * 1.0 and cur["ctx"][model]:
            return self.layout
        if model == "qwen":
            order = ["boost", "long"] if self.mode == "boost" else ["shared", "boost", "long"]
        else:
            order = (["inkboost", "ink64", "ink128"] if self.mode == "inkboost"
                     else ["shared", "inkboost", "ink64", "ink128"])
        fits = [l for l in order if LAYOUTS[l]["ctx"][model] >= need and self._allowed(l, now)]
        if fits:
            return fits[0]
        # Too big for anything allowed: the roomiest allowed layout, and let
        # llama.cpp say the prompt doesn't fit.
        ok = [l for l in order if self._allowed(l, now)] or ["shared"]
        return max(ok, key=lambda l: LAYOUTS[l]["ctx"][model])

    async def _after(self, model):
        # Between requests is the only time a restart is cheap. A follow-up
        # request that lands mid-switch waits at the gate.
        await asyncio.sleep(0.5)
        if self.inflight[model] or self.switching or self.layout != "shared":
            return
        target = BOOST_LAYOUT[model]
        other = "inkling" if model == "qwen" else "qwen"
        now = time.time()
        if self.mode != "auto" or not self._allowed(target, now):
            return      # pinned modes are handled by reconcile
        if sum(1 for t in self.recent[model] if now - t < BOOST_WINDOW) < BOOST_AFTER:
            return
        if now - self.last_req[other] < GRACE or self.inflight[other]:
            return
        if await self._busy_direct(other):
            return      # e.g. Home Assistant mid-reply on Qwen
        if await self._comfy_busy():
            return
        name = "Qwen" if model == "qwen" else "Inkling"
        await self.switch(target, f"{name} session")

    async def reconcile(self):
        up = await self.running()
        now = time.time()
        match = [n for n in LAYOUTS if self._targets(n) and set(self._targets(n)) == up]
        if match:
            self.layout = match[0]
        elif not up:
            self.layout = "shared"
        lay = LAYOUTS[self.layout]
        rest = await self._rest(now)

        target = None
        if self.layout != rest and lay["owner"] is None:
            target = rest       # a model was parked: the survivor gets the GPU
        elif self.layout != rest:
            owner = lay["owner"]
            idle = (now - self.last_req[owner] > lay["restore"]) and not self.inflight[owner]
            pin = self.mode if self.mode in BOOST_LAYOUT.values() else None
            if pin == self.layout:
                idle = False
            if (idle or not self._allowed(self.layout, now)
                    or (lay["solo"] and await self._comfy_busy())):
                if not await self._busy_direct(owner):
                    # A pinned boost is where a long/big layout falls back to.
                    target = (pin if pin and LAYOUTS[pin]["owner"] == owner
                              and self._allowed(pin, now) else rest)
        pin = self.mode if self.mode in BOOST_LAYOUT.values() else None
        if target is None and pin and self.layout == "shared" \
                and self._allowed(pin, now) and not await self._comfy_busy():
            # Give the other model a minute after it was used, so a pinned
            # boost doesn't bounce while someone is still talking to it.
            other = "inkling" if LAYOUTS[pin]["owner"] == "qwen" else "qwen"
            if now - self.last_req[other] > 60 and not self.inflight[other] \
                    and not await self._busy_direct(other):
                target = pin
        if target is None and not match:
            target = rest if self.layout in ("shared", rest) or not up else self.layout
        if target:
            await self.switch(target, "idle" if target == rest and lay["owner"]
                              else (f"pinned {LAYOUTS[target]['label']}" if target == pin
                                    else "repair"))

    async def loop(self):
        await asyncio.sleep(3)
        self.event("director started")
        while True:
            try:
                if self._settled.is_set():
                    await self.reconcile()
            except Exception as exc:
                log.warning("director reconcile: %s", exc)
            await asyncio.sleep(10)

    def start(self):
        self._task = asyncio.create_task(self.loop())

    async def aclose(self):
        if self._task:
            self._task.cancel()
        for c in (self.docker, self.http, self.probe):
            await c.aclose()

    # ---------------------------------------------------------------- control

    async def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError("mode must be one of " + ", ".join(MODES))
        if mode == "inkboost" and not self._allowed("inkboost", time.time()) \
                and ("inkling" in self.parked or not ALLOW_QWEN_STOP):
            raise ValueError("Inkling boost needs Inkling started and DIRECTOR_ALLOW_QWEN_STOP=1")
        self.mode = mode
        self._save()
        self.event(f"mode set to {mode}")
        if mode == "shared" and self.layout != await self._rest(time.time()):
            asyncio.create_task(self.switch(await self._rest(time.time()), "pinned both"))
        elif mode in BOOST_LAYOUT.values() and self.layout == "shared":
            asyncio.create_task(self.switch(mode, f"pinned {LAYOUTS[mode]['label']}"))

    async def container_action(self, name, action):
        """Dashboard start/stop/restart of a model container, kept consistent:
        Stop parks the model (it stays off, and the other one gets the GPU),
        Start unparks it, Restart restarts whichever variant is running."""
        model = CONTAINERS[name][0]
        label = "Qwen" if model == "qwen" else "Inkling"
        if action == "stop":
            self.parked.add(model)
            self._save()
            self.event(f"{label} stopped by hand (stays off until started)")
            await self.switch(await self._rest(time.time()), f"{label} stopped by hand")
            return
        if action == "start":
            if model not in self.parked:
                raise ValueError(f"{label} isn't stopped; the director starts its variants")
            self.parked.discard(model)
            self._save()
            self.event(f"{label} started by hand")
            await self.switch(await self._rest(time.time()), f"{label} started by hand")
            return
        if await self.cstate(name) != "running":
            raise ValueError(f"{name} isn't running")
        await self.cact(name, "restart")
        self.event(f"{name} restarted by hand")

    def snapshot(self):
        now = time.time()
        lay = LAYOUTS[self.layout]
        return {
            "enabled": True,
            "mode": self.mode,
            "layout": self.layout,
            "layout_label": (lay["label"] if self.layout != "shared" or not self.parked
                             else ("Both stopped" if len(self.parked) == len(MODELS)
                                   else ("Inkling only" if "qwen" in self.parked
                                         else "Qwen only"))),
            "layout_detail": (lay["detail"] if self.layout != "shared" or not self.parked
                              else "at its shared size while ComfyUI is busy or in the "
                                   "nightly window; it gets the fast layout after"),
            "ctx": lay["ctx"],
            "switching": self.switching,
            "switching_label": LAYOUTS[self.switching]["label"] if self.switching else None,
            "quiet": _in_quiet(now),
            "quiet_window": QUIET,
            "blocked_for": max(0, int(self.block_until - now)),
            "parked": sorted(self.parked),
            "allow_qwen_stop": ALLOW_QWEN_STOP,
            "managed": sorted(MANAGED),
            "models": {
                m: {"inflight": self.inflight[m],
                    "last_request_ago": int(now - self.last_req[m]) if self.last_req[m] else None,
                    "recent": sum(1 for t in self.recent[m] if now - t < BOOST_WINDOW)}
                for m in MODELS},
            "layouts": {k: {"label": v["label"], "detail": v["detail"], "ctx": v["ctx"]}
                        for k, v in LAYOUTS.items()},
            "policy": {"boost_after": BOOST_AFTER, "window_s": BOOST_WINDOW,
                       "grace_s": GRACE},
            "events": list(self.events)[-20:],
            "stats": self.stats,
        }

    # ------------------------------------------------------------------ proxy

    async def _gate(self, model, raw: bytes):
        t0 = time.time()
        await self._settled.wait()
        if model in self.parked:
            return f"{model} is stopped (start it from Sparkboard)"
        target = self._pick(model, _need_tokens(raw), time.time())
        if target != self.layout:
            await self.switch(target, f"{model} needs it")
        await self._settled.wait()
        if not LAYOUTS[self.layout]["ctx"][model]:
            return f"{model} is not loaded and could not be started"
        self.stats["held_ms_total"] += int((time.time() - t0) * 1000)
        return None

    async def forward(self, request, model: str, path: str):
        from fastapi.responses import JSONResponse, Response, StreamingResponse

        if model not in MODELS:
            return JSONResponse({"error": f"unknown model {model}"}, status_code=404)
        if API_KEY:
            import hmac
            given = request.headers.get("authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {API_KEY}".encode()):
                # Checked here, not just upstream, so a stranger's request
                # can't count toward a session or trigger a layout change.
                return JSONResponse({"error": {"code": 401, "message": "Invalid API Key",
                                               "type": "authentication_error"}}, status_code=401)
        raw = await request.body()
        is_gen = request.method == "POST" and bool(raw)
        if is_gen:
            err = await self._gate(model, raw)
            if err:
                return JSONResponse({"error": err}, status_code=503)
            now = time.time()
            self.last_req[model] = now
            self.recent[model].append(now)
        elif not LAYOUTS[self.layout]["ctx"][model]:
            if path.rstrip("/").endswith("models"):
                # opencode lists models before asking; answer even while parked.
                return JSONResponse({"object": "list", "data": [
                    {"id": model, "object": "model", "owned_by": "sparkboard-director"}]})

        url = f"{UPSTREAM[model]}/{path.lstrip('/')}"
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length", "connection")}
        stream = False
        if raw:
            try:
                stream = bool(json.loads(raw).get("stream"))
            except (ValueError, AttributeError):
                pass

        self.stats["forwarded"] += 1
        if is_gen:
            self.inflight[model] += 1
        finished = False

        def done():
            nonlocal finished
            if not is_gen or finished:
                return
            finished = True
            self.inflight[model] -= 1
            self.last_req[model] = time.time()
            asyncio.create_task(self._after(model))

        if not stream:
            try:
                up = await self.http.request(request.method, url, content=raw,
                                             headers=headers, params=request.query_params)
            except httpx.HTTPError as exc:
                done()
                return JSONResponse({"error": f"upstream unreachable: {exc}"}, status_code=502)
            done()
            passthru = {k: v for k, v in up.headers.items()
                        if k.lower() not in ("content-length", "transfer-encoding",
                                             "connection", "content-encoding")}
            return Response(content=up.content, status_code=up.status_code,
                            headers=passthru, media_type=up.headers.get("content-type"))

        try:
            resp = await self.http.send(
                self.http.build_request(request.method, url, content=raw,
                                        headers=headers, params=request.query_params),
                stream=True)
        except httpx.HTTPError as exc:
            done()
            return JSONResponse({"error": f"upstream unreachable: {exc}"}, status_code=502)

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                done()

        return StreamingResponse(relay(), status_code=resp.status_code,
                                 media_type=resp.headers.get("content-type", "text/event-stream"),
                                 headers={"Cache-Control": "no-cache, no-transform",
                                          "X-Accel-Buffering": "no"})
