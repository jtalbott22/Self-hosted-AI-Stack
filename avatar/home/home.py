"""home: lets the avatar control the house through Home Assistant.

The avatar sends a plain-English command ("turn off the kitchen lights",
"is the front door locked?") and this passes it to Home Assistant's own
assistant (Assist, POST /api/conversation/process), which works out the
devices and does it. The Home Assistant token stays here, never in the page.

Only the owner may use it. nginx says who's asking in X-Avatar-User (the
Google account from signing in; set or cleared on every request, so a
browser can't choose it). Empty means the LAN, which needs no sign-in and
is allowed. Anyone else gets 403.

  GET  /ok        {"allowed", "ready"}: whether the avatar should offer it
  POST /command   {"text"} -> {"ok", "reply", "kind", "done", "ids"}
  GET  /devices   ?ids=light.a,switch.b or ?q=basement lights: their states,
                  for the avatar's controls panel
  POST /set       {"id", "action", "value", "quick"}: one control, from that
                  panel. "quick" (while a slider is being dragged) sends it
                  and returns at once, without reading the device back.
                  Only fixed actions per kind of device; locks can be locked
                  but never unlocked from here.

Settings (environment): HA_URL (http://host.docker.internal:8123),
HA_TOKEN (a long-lived access token from your Home Assistant profile),
OWNERS (comma-separated Google accounts), PORT (8707).
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HA_URL = os.environ.get("HA_URL", "http://host.docker.internal:8123").rstrip("/")
HA_TOKEN = os.environ.get("HA_TOKEN", "").strip()
OWNERS = {e.strip().lower() for e in os.environ.get("OWNERS", "").split(",") if e.strip()}
PORT = int(os.environ.get("PORT", "8707"))
DEVICE_DOMAINS = ("light", "switch", "fan", "climate", "cover", "lock", "media_player", "scene", "script", "vacuum")


def ha(method, path, body=None, timeout=20):
    req = urllib.request.Request(HA_URL + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Bearer " + HA_TOKEN, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


_areas = {"at": 0, "map": {}}
_areas_lock = threading.Lock()


def areas():
    """entity_id -> area name (the rooms set up in Home Assistant), kept a minute."""
    with _areas_lock:
        if time.time() - _areas["at"] > 60:
            try:
                tpl = "{% for s in states %}{{ s.entity_id }}|{{ area_name(s.entity_id) or '' }}\n{% endfor %}"
                req = urllib.request.Request(HA_URL + "/api/template", method="POST",
                                             data=json.dumps({"template": tpl}).encode(),
                                             headers={"Authorization": "Bearer " + HA_TOKEN, "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    text = r.read().decode()
                _areas["map"] = dict(line.split("|", 1) for line in text.splitlines() if "|" in line)
            except Exception as e:
                print(f"home: couldn't read areas: {e}", flush=True)
            _areas["at"] = time.time()
        return _areas["map"]


def describe(s, area=""):
    """One device, as the controls panel needs it."""
    eid, a, st = s["entity_id"], s["attributes"], s["state"]
    domain = eid.split(".")[0]
    d = {"id": eid, "domain": domain, "name": a.get("friendly_name") or eid, "state": st, "area": area}
    if domain == "light":
        modes = a.get("supported_color_modes") or []
        d["dimmable"] = any(m != "onoff" for m in modes)
        d["brightness"] = round((a.get("brightness") or 0) / 2.55) if st == "on" else 0
    elif domain == "fan" and a.get("percentage") is not None:
        d["percentage"] = a.get("percentage")
    elif domain == "cover" and a.get("current_position") is not None:
        d["position"] = a.get("current_position")
    elif domain == "media_player":
        d["volume"] = round((a.get("volume_level") or 0) * 100) if a.get("volume_level") is not None else None
        d["muted"] = a.get("is_volume_muted")
        d["title"] = " - ".join(x for x in (a.get("media_artist"), a.get("media_title")) if x)
        d["app"] = a.get("app_name") or ""
        d["tv"] = a.get("device_class") == "tv"
        f = a.get("supported_features") or 0
        d["can"] = [n for bit, n in MEDIA_FEATURES if f & bit]
    elif domain == "climate":
        d["current"] = a.get("current_temperature")
        d["target"] = a.get("temperature")
        d["unit"] = a.get("temperature_unit") or ""
        d["step"] = a.get("target_temp_step") or 1
    return d


MEDIA_FEATURES = ((1, "pause"), (4, "volume_set"), (8, "mute"), (16, "previous"), (32, "next"),
                  (128, "on"), (256, "off"), (1024, "volume_step"), (16384, "play"))

# What the panel may do: (domain, action) -> (service, how the value goes in)
ACTIONS = {
    ("light", "on"): ("turn_on", None), ("light", "off"): ("turn_off", None),
    ("light", "brightness"): ("turn_on", lambda v: {"brightness_pct": max(1, min(100, int(v)))}),
    ("switch", "on"): ("turn_on", None), ("switch", "off"): ("turn_off", None),
    ("fan", "on"): ("turn_on", None), ("fan", "off"): ("turn_off", None),
    ("fan", "percentage"): ("set_percentage", lambda v: {"percentage": max(0, min(100, int(v)))}),
    ("cover", "open"): ("open_cover", None), ("cover", "close"): ("close_cover", None), ("cover", "stop"): ("stop_cover", None),
    ("cover", "position"): ("set_cover_position", lambda v: {"position": max(0, min(100, int(v)))}),
    ("media_player", "on"): ("turn_on", None), ("media_player", "off"): ("turn_off", None),
    ("media_player", "play_pause"): ("media_play_pause", None),
    ("media_player", "next"): ("media_next_track", None), ("media_player", "previous"): ("media_previous_track", None),
    ("media_player", "volume_up"): ("volume_up", None), ("media_player", "volume_down"): ("volume_down", None),
    ("media_player", "mute"): ("volume_mute", lambda v: {"is_volume_muted": bool(v)}),
    ("media_player", "volume"): ("volume_set", lambda v: {"volume_level": max(0, min(100, int(v))) / 100}),
    ("scene", "activate"): ("turn_on", None), ("script", "run"): ("turn_on", None),
    ("climate", "target"): ("set_temperature", lambda v: {"temperature": float(v)}),
    ("lock", "lock"): ("lock", None),
    ("vacuum", "start"): ("start", None), ("vacuum", "home"): ("return_to_base", None),
}


def device_names():
    """What there is to control, for when Assist didn't recognise a name."""
    try:
        states = ha("GET", "/api/states")
    except Exception:
        return []
    out = []
    for s in states:
        domain = s["entity_id"].split(".")[0]
        if domain in DEVICE_DOMAINS:
            out.append(f'{s["attributes"].get("friendly_name") or s["entity_id"]} ({domain}, {s["state"]})')
    return sorted(out)[:80]


class Handler(BaseHTTPRequestHandler):
    def allowed(self):
        u = (self.headers.get("X-Avatar-User") or "").strip().lower()
        return not u or u in OWNERS

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def devices(self):
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        ids = [i for i in (qs.get("ids", [""])[0]).split(",") if i]
        words = [w for w in re.split(r"\W+", (qs.get("q", [""])[0]).lower()) if w and w not in ("the", "all", "and", "in", "my", "of")]
        try:
            states = ha("GET", "/api/states")
        except Exception as e:
            return self.reply(502, {"error": f"Home Assistant didn't answer ({e})"})
        amap = areas()
        out = []
        for s in states:
            eid = s["entity_id"]
            if eid.split(".")[0] not in DEVICE_DOMAINS:
                continue
            area = amap.get(eid, "")
            if ids and eid not in ids:
                continue
            if words:
                hay = " ".join((eid.replace("_", " "), (s["attributes"].get("friendly_name") or ""), area)).lower()
                # "lights" finds lights, "tv" the TVs and speakers, as well as names
                hay += {"light": " light lights lamp lamps", "switch": " switch switches light lights",
                        "media_player": " tv tvs speaker speakers media", "climate": " thermostat temperature heat",
                        "scene": " scene scenes", "cover": " blinds shades", "fan": " fan fans"}.get(eid.split(".")[0], "")
                if not all(w.rstrip("s") in hay for w in words):
                    continue
            out.append(describe(s, area))
        if ids:
            out.sort(key=lambda d: ids.index(d["id"]))
        else:
            out.sort(key=lambda d: (d["area"], d["domain"], d["name"]))
        self.reply(200, {"devices": out[:60]})

    def do_GET(self):
        if self.path.split("?")[0] == "/devices":
            if not self.allowed():
                return self.reply(403, {"error": "home control is only for the owner"})
            return self.devices()
        if self.path.split("?")[0] != "/ok":
            return self.reply(404, {"error": "not found"})
        if not self.allowed():
            return self.reply(403, {"allowed": False})
        if not HA_TOKEN:
            return self.reply(200, {"allowed": True, "ready": False, "error": "HA_TOKEN isn't set"})
        try:
            ha("GET", "/api/", timeout=5)
            return self.reply(200, {"allowed": True, "ready": True})
        except urllib.error.HTTPError as e:
            return self.reply(200, {"allowed": True, "ready": False, "error": f"Home Assistant said {e.code}"})
        except Exception as e:
            return self.reply(200, {"allowed": True, "ready": False, "error": f"can't reach Home Assistant ({e})"})

    def set(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            b = json.loads(self.rfile.read(min(n, 10000)) or b"{}")
            eid, action, value = str(b.get("id") or ""), str(b.get("action") or ""), b.get("value")
            quick = bool(b.get("quick"))
        except Exception:
            return self.reply(400, {"ok": False, "error": "bad request"})
        domain = eid.split(".")[0]
        spec = ACTIONS.get((domain, action))
        if not spec or not re.fullmatch(r"[a-z_]+\.[a-z0-9_]+", eid):
            return self.reply(400, {"ok": False, "error": f"can't {action} a {domain}"})
        service, conv = spec
        data = {"entity_id": eid}
        try:
            if conv:
                data.update(conv(value))
            ha("POST", f"/api/services/{domain}/{service}", data)
            if quick:
                return self.reply(200, {"ok": True})
            time.sleep(0.4)                            # let the device report back
            s = ha("GET", f"/api/states/{eid}")
        except Exception as e:
            return self.reply(502, {"ok": False, "error": f"Home Assistant didn't do it ({e})"})
        print(f"home: panel {domain}.{service} {eid}" + (f" {value}" if value is not None else ""), flush=True)
        self.reply(200, {"ok": True, "device": describe(s, areas().get(eid, ""))})

    def do_POST(self):
        if self.path.split("?")[0] == "/set":
            if not self.allowed():
                return self.reply(403, {"ok": False, "error": "home control is only for the owner"})
            return self.set()
        if self.path.split("?")[0] != "/command":
            return self.reply(404, {"error": "not found"})
        if not self.allowed():
            return self.reply(403, {"ok": False, "error": "home control is only for the owner"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            text = str(json.loads(self.rfile.read(min(n, 10000)) or b"{}").get("text") or "").strip()[:500]
        except Exception:
            text = ""
        if not text:
            return self.reply(400, {"ok": False, "error": "no command"})
        try:
            r = ha("POST", "/api/conversation/process", {"text": text, "language": "en"})
        except Exception as e:
            print(f"home: Home Assistant failed: {e}", flush=True)
            return self.reply(502, {"ok": False, "error": f"Home Assistant didn't answer ({e})"})
        resp = (r or {}).get("response") or {}
        speech = ((resp.get("speech") or {}).get("plain") or {}).get("speech") or ""
        kind = resp.get("response_type") or ""
        hits = [t for t in ((resp.get("data") or {}).get("success") or []) if t.get("type") == "entity"]
        done = [t.get("name") for t in hits if t.get("name")]
        out = {"ok": kind != "error", "reply": speech, "kind": kind, "done": done,
               "ids": [t["id"] for t in hits if t.get("id")][:40]}
        if kind == "error":
            out["code"] = (resp.get("data") or {}).get("code")
            out["devices"] = device_names()
        print(f"home: {kind or 'no answer'}" + (f" ({', '.join(done)})" if done else ""), flush=True)
        self.reply(200, out)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"home: Home Assistant at {HA_URL}; token {'set' if HA_TOKEN else 'NOT set'}; "
          f"owners: {', '.join(sorted(OWNERS)) or 'none (LAN only)'}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
