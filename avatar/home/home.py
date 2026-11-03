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
  POST /command   {"text"} -> {"ok", "reply", "kind", "done"}

Settings (environment): HA_URL (http://host.docker.internal:8123),
HA_TOKEN (a long-lived access token from your Home Assistant profile),
OWNERS (comma-separated Google accounts), PORT (8707).
"""
import json
import os
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

    def do_GET(self):
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

    def do_POST(self):
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
        done = [t.get("name") for t in ((resp.get("data") or {}).get("success") or []) if t.get("name")]
        out = {"ok": kind != "error", "reply": speech, "kind": kind, "done": done}
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
