"""authgate: fail2ban for nginx basic auth, without reloading nginx.

It follows nginx's error log for failed logins ("password mismatch", or
"user ... was not found"), counts them per client address, and bans an
address that fails too often. nginx asks it about every request with
auth_request (GET /check, the client's address in X-Real-IP): 403 for a
banned address, 204 otherwise. Bans start and end at once, with no reload,
which matters where nginx runs as a Windows service that only an admin can
signal.

Behind a CDN, nginx must log the visitor's address, not the CDN's: see
avatar/nginx-authgate.conf.

Also: GET /bans lists them, POST /unban?ip=1.2.3.4 lifts one.
Settings (environment): ERROR_LOG, FAILS (5), WINDOW (900 s), BAN (3600 s),
REPEAT_BAN (86400 s, for an address banned again within a day), STATE
(/data/bans.json), PORT (8705).
"""
import ipaddress
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

LOG = os.environ.get("ERROR_LOG", "/logs/error.log")
FAILS = int(os.environ.get("FAILS", "5"))
WINDOW = int(os.environ.get("WINDOW", "900"))
BAN = int(os.environ.get("BAN", "3600"))
REPEAT_BAN = int(os.environ.get("REPEAT_BAN", "86400"))
STATE = os.environ.get("STATE", "/data/bans.json")
PORT = int(os.environ.get("PORT", "8705"))

FAILED = re.compile(r'user "([^"]*)":? (?:password mismatch|was not found in [^,]*), client: ([0-9a-fA-F:.]+)')

lock = threading.Lock()
fails: dict[str, list[float]] = {}          # address -> times of recent failures
bans: dict[str, dict] = {}                  # address -> {"until", "count", "last"}


LOCAL = [ipaddress.ip_network(n) for n in
         ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8", "169.254.0.0/16",
          "::1/128", "fc00::/7", "fe80::/10")]


def private(ip: str) -> bool:
    """The LAN and this machine are never banned."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return any(a in n for n in LOCAL)


def load():
    try:
        with open(STATE) as f:
            bans.update(json.load(f))
    except (OSError, ValueError):
        pass


def save():
    try:
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        tmp = STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(bans, f)
        os.replace(tmp, STATE)
    except OSError as e:
        print("authgate: couldn't save bans:", e, flush=True)


def failed(ip: str, user: str):
    if private(ip):
        return
    now = time.time()
    with lock:
        recent = [t for t in fails.get(ip, []) if now - t < WINDOW] + [now]
        fails[ip] = recent
        print(f"authgate: failed login {len(recent)}/{FAILS} from {ip} (user {user!r})", flush=True)
        if len(recent) < FAILS:
            return
        old = bans.get(ip, {})
        again = old and now - old.get("last", 0) < 86400
        length = REPEAT_BAN if again else BAN
        bans[ip] = {"until": now + length, "count": old.get("count", 0) + 1, "last": now}
        fails.pop(ip, None)
        save()
    print(f"authgate: BANNED {ip} for {length // 60} min after {FAILS} failed logins"
          + (" (again today)" if again else ""), flush=True)


def banned(ip: str) -> bool:
    b = bans.get(ip)
    return bool(b) and b["until"] > time.time()


def follow():
    """Read new lines as nginx writes them. Starts at the end: old failures
    don't count. Copes with the log being rotated or truncated."""
    pos = None
    while True:
        try:
            size = os.path.getsize(LOG)
            if pos is None or size < pos:
                pos = size if pos is None else 0
            if size > pos:
                with open(LOG, "rb") as f:
                    f.seek(pos)
                    chunk = f.read(size - pos)
                end = chunk.rfind(b"\n") + 1          # leave a half-written line for next time
                pos += end
                for line in chunk[:end].decode("utf-8", "replace").splitlines():
                    m = FAILED.search(line)
                    if m:
                        failed(m.group(2), m.group(1))
        except FileNotFoundError:
            pos = None
        except OSError as e:
            print("authgate: reading the log:", e, flush=True)
        time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body=None):
        data = json.dumps(body, indent=1).encode() if body is not None else b""
        self.send_response(code)
        if data:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/check":
            ip = (self.headers.get("X-Real-IP") or "").strip()
            return self.reply(403 if banned(ip) else 204)
        if u.path == "/bans":
            now = time.time()
            return self.reply(200, {ip: {"minutes_left": round((b["until"] - now) / 60), "times": b["count"]}
                                    for ip, b in bans.items() if b["until"] > now})
        if u.path == "/health":
            return self.reply(200, {"ok": True, "log": LOG, "log_found": os.path.exists(LOG)})
        self.reply(404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/unban":
            ip = parse_qs(u.query).get("ip", [""])[0]
            with lock:
                gone = bans.pop(ip, None)
                fails.pop(ip, None)
                save()
            print(f"authgate: unbanned {ip}" if gone else f"authgate: {ip} wasn't banned", flush=True)
            return self.reply(200, {"unbanned": bool(gone)})
        self.reply(404)

    def log_message(self, *a):                 # /check runs on every request; keep the log for bans
        pass


if __name__ == "__main__":
    load()
    print(f"authgate: following {LOG}; {FAILS} failures in {WINDOW // 60} min bans for {BAN // 60} min "
          f"({REPEAT_BAN // 3600} h if again within a day)", flush=True)
    threading.Thread(target=follow, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
