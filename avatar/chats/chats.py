"""chats: keeps the avatar's conversations, separately for each person.

nginx says who's asking in X-Avatar-User: the Google account from signing in
(avatar-signin, oauth2-proxy). nginx sets or clears that header on every
request, so a browser can't pick someone else's. A request without it came
from the LAN, which nginx lets in without signing in; it's filed under
LAN_USER.

  GET    /me           who this is: {"user", "lan"}
  GET    /chats        that person's chats, newest first (no messages)
  GET    /chats/<id>   one chat, with its messages
  PUT    /chats/<id>   save it (JSON: title, messages, turns); creates it
  DELETE /chats/<id>   delete one
  DELETE /chats        delete all of them

Each person's chats are JSON files in /data/<hash of the email>/.
Settings (environment): LAN_USER ("lan"), DATA (/data), PORT (8706),
MAX_BYTES (8 MB per chat).
"""
import hashlib
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DATA = os.environ.get("DATA", "/data")
LAN_USER = os.environ.get("LAN_USER", "lan").strip().lower() or "lan"
PORT = int(os.environ.get("PORT", "8706"))
MAX_BYTES = int(os.environ.get("MAX_BYTES", str(8 * 1024 * 1024)))
ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

locks: dict[str, threading.Lock] = {}
locks_lock = threading.Lock()


def lock_for(user: str) -> threading.Lock:
    with locks_lock:
        return locks.setdefault(user, threading.Lock())


def folder(user: str) -> str:
    return os.path.join(DATA, hashlib.sha256(user.encode()).hexdigest()[:32])


def read(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


class Handler(BaseHTTPRequestHandler):
    def who(self):
        u = (self.headers.get("X-Avatar-User") or "").strip().lower()
        return (u, False) if u else (LAN_USER, True)

    def reply(self, code, body=None):
        data = json.dumps(body).encode() if body is not None else b""
        self.send_response(code)
        self.send_header("Cache-Control", "no-store")
        if data:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def route(self):
        p = self.path.split("?", 1)[0].rstrip("/")
        if p == "/chats":
            return None, True
        m = re.fullmatch(r"/chats/([^/]+)", p)
        if m and ID.match(m.group(1)):
            return m.group(1), True
        return None, False

    def do_GET(self):
        user, lan = self.who()
        if self.path.split("?")[0] == "/me":
            return self.reply(200, {"user": user, "lan": lan})
        cid, ok = self.route()
        if not ok:
            return self.reply(404)
        d = folder(user)
        if cid:
            c = read(os.path.join(d, cid + ".json"))
            return self.reply(200, c) if c else self.reply(404)
        out = []
        try:
            names = os.listdir(d)
        except FileNotFoundError:
            names = []
        for n in names:
            if n.endswith(".json"):
                c = read(os.path.join(d, n))
                if c:
                    out.append({k: c.get(k) for k in ("id", "title", "created", "updated")}
                               | {"turns": len(c.get("turns") or [])})
        out.sort(key=lambda c: c.get("updated") or 0, reverse=True)
        self.reply(200, out)

    def do_PUT(self):
        user, _ = self.who()
        cid, ok = self.route()
        if not ok or not cid:
            return self.reply(404)
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BYTES:
            return self.reply(413, {"error": "too big"})
        try:
            body = json.loads(self.rfile.read(n))
            assert isinstance(body, dict) and isinstance(body.get("messages"), list)
        except Exception:
            return self.reply(400, {"error": "expected JSON with messages"})
        path = os.path.join(folder(user), cid + ".json")
        now = time.time()
        with lock_for(user):
            old = read(path) or {}
            c = {"id": cid, "title": str(body.get("title") or "Chat")[:120],
                 "created": old.get("created") or now, "updated": now,
                 "messages": body["messages"], "turns": body.get("turns") or []}
            write(path, c)
        self.reply(200, {"id": cid, "updated": now})

    def do_DELETE(self):
        user, _ = self.who()
        cid, ok = self.route()
        if not ok:
            return self.reply(404)
        d = folder(user)
        gone = 0
        with lock_for(user):
            names = [cid + ".json"] if cid else (os.listdir(d) if os.path.isdir(d) else [])
            for n in names:
                if n.endswith(".json"):
                    try:
                        os.remove(os.path.join(d, n))
                        gone += 1
                    except FileNotFoundError:
                        pass
        print(f"chats: {user} deleted {gone} chat(s)", flush=True)
        self.reply(200, {"deleted": gone})

    def log_message(self, *a):             # chats are private; don't log who said what
        pass


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    print(f"chats: keeping chats in {DATA}; LAN requests are filed as {LAN_USER}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
