#!/usr/bin/env bash
#
# Sparkboard installer.
#
#   sudo ./install.sh              install or upgrade
#   sudo ./install.sh --uninstall  remove service, keep collected data
#   sudo ./install.sh --purge      remove everything including the database
#
# Safe to re-run: an upgrade reuses the existing database and user.

set -euo pipefail

APP_DIR=/opt/sparkboard
DATA_DIR=/var/lib/sparkboard
SVC_USER=sparkboard
SVC_NAME=sparkboard
PORT="${SPARKBOARD_PORT:-9101}"
BIND="${SPARKBOARD_BIND:-127.0.0.1}"
INTERVAL="${SPARKBOARD_INTERVAL:-2}"
# vLLM activity feed (off by default). Enable with SPARKBOARD_PROXY=1.
PROXY="${SPARKBOARD_PROXY:-0}"
PROXY_TRANSPARENT="${SPARKBOARD_PROXY_TRANSPARENT:-0}"
VLLM_UPSTREAM="${SPARKBOARD_VLLM_UPSTREAM:-http://127.0.0.1:8000}"
CLASSIFY_UPSTREAM="${SPARKBOARD_CLASSIFY_UPSTREAM:-}"
CLASSIFY_MODEL="${SPARKBOARD_CLASSIFY_MODEL:-}"
VLLM_API_KEY="${SPARKBOARD_VLLM_API_KEY:-}"
INJECT_AUTH="${SPARKBOARD_INJECT_AUTH:-0}"
CLASSIFY_SAMPLE="${SPARKBOARD_CLASSIFY_SAMPLE:-1.0}"
CLASSIFY_CONCURRENCY="${SPARKBOARD_CLASSIFY_CONCURRENCY:-2}"
NGINX_SNIPPET=/etc/nginx/snippets/sparkboard.conf
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bold=$'\033[1m'; dim=$'\033[2m'; grn=$'\033[32m'; ylw=$'\033[33m'; red=$'\033[31m'; off=$'\033[0m'
say()  { printf '%s\n' "$*"; }
step() { printf '%s==>%s %s\n' "$bold" "$off" "$*"; }
ok()   { printf '  %s+%s %s\n' "$grn" "$off" "$*"; }
warn() { printf '  %s!%s %s\n' "$ylw" "$off" "$*"; }
die()  { printf '  %sx%s %s\n' "$red" "$off" "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo"

# ------------------------------------------------------------- uninstall

if [[ "${1:-}" == "--uninstall" || "${1:-}" == "--purge" ]]; then
  step "Removing Sparkboard"
  systemctl stop  "$SVC_NAME" 2>/dev/null || true
  systemctl disable "$SVC_NAME" 2>/dev/null || true
  rm -f "/etc/systemd/system/${SVC_NAME}.service" "$NGINX_SNIPPET"
  systemctl daemon-reload
  rm -rf "$APP_DIR"
  ok "service and application files removed"
  if [[ "${1:-}" == "--purge" ]]; then
    rm -rf "$DATA_DIR"
    userdel "$SVC_USER" 2>/dev/null || true
    ok "database and service user removed"
  else
    say "  ${dim}history kept at $DATA_DIR (use --purge to delete)${off}"
  fi
  say ""
  say "Remember to drop the include line from your nginx config and reload."
  exit 0
fi

# ------------------------------------------------------------ preflight

step "Checking the machine"

command -v python3 >/dev/null || die "python3 not found"
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || die "python 3.10+ required, found $PYV"
ok "python $PYV"

if ! python3 -c 'import venv' 2>/dev/null; then
  warn "python venv module missing, installing python3-venv"
  apt-get update -qq && apt-get install -y -qq python3-venv \
    || die "could not install python3-venv, install it and re-run"
fi

if command -v nvidia-smi >/dev/null; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
  ok "nvidia-smi found${GPU:+ -- $GPU}"
else
  warn "nvidia-smi not on PATH; Sparkboard will show host metrics only"
fi

DOCKER_OK=0
if command -v docker >/dev/null && getent group docker >/dev/null; then
  if [[ "${SPARKBOARD_DOCKER:-1}" == "1" ]]; then
    DOCKER_OK=1
    ok "docker found -- container stats will be enabled"
  else
    warn "docker found but SPARKBOARD_DOCKER=0, skipping container stats"
  fi
elif command -v docker >/dev/null; then
  warn "docker found but no 'docker' group; container stats disabled"
fi

if ss -ltn 2>/dev/null | grep -q ":${PORT}[[:space:]]" \
   && ! systemctl is-active --quiet "$SVC_NAME"; then
  die "port $PORT is already in use -- re-run with SPARKBOARD_PORT=9xxx sudo ./install.sh"
fi
ok "port $PORT available"

if [[ "$PROXY" == "1" ]]; then
  # The activity feed proxies vLLM. Probe the upstream now so a typo in the
  # URL surfaces here rather than as a dead feed after install.
  if curl -fsS -m 3 "$VLLM_UPSTREAM/v1/models" >/dev/null 2>&1; then
    ok "vLLM reachable at $VLLM_UPSTREAM -- activity feed enabled"
  else
    warn "vLLM not reachable at $VLLM_UPSTREAM"
    warn "the feed will start empty and recover once vLLM is up; if the URL is"
    warn "wrong, re-run with SPARKBOARD_VLLM_UPSTREAM=http://host:port sudo ./install.sh"
  fi
fi

# -------------------------------------------------------------- install

step "Installing to $APP_DIR"

if ! id "$SVC_USER" >/dev/null 2>&1; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
  ok "created service user $SVC_USER"
else
  ok "service user $SVC_USER exists"
fi

if [[ $DOCKER_OK -eq 1 ]]; then
  if ! id -nG "$SVC_USER" | tr ' ' '\n' | grep -qx docker; then
    usermod -aG docker "$SVC_USER"
    warn "added $SVC_USER to the docker group -- this is effectively root on"
    warn "this host, since docker can mount any path into a container."
    warn "Re-run with SPARKBOARD_DOCKER=0 to skip container stats instead."
  else
    ok "$SVC_USER already in the docker group"
  fi
fi

install -d -m 755 "$APP_DIR"
install -d -m 750 -o "$SVC_USER" -g "$SVC_USER" "$DATA_DIR"

rm -rf "$APP_DIR/app" "$APP_DIR/static"
cp -r "$SRC/app" "$SRC/static" "$APP_DIR/"
cp "$SRC/requirements.txt" "$APP_DIR/"
chmod -R a+rX "$APP_DIR"
ok "application files copied"

if [[ ! -x "$APP_DIR/venv/bin/python" ]]; then
  python3 -m venv "$APP_DIR/venv"
  ok "virtualenv created"
fi
"$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
ok "dependencies installed"

# --------------------------------------------------------------- systemd

step "Installing the service"

DOCKER_UNIT=""
if [[ $DOCKER_OK -eq 1 ]]; then
  DOCKER_UNIT="SupplementaryGroups=docker
# ProtectSystem=strict would otherwise leave the docker socket read-only.
ReadWritePaths=-/run/docker.sock"
fi

PROXY_UNIT=""
if [[ "$PROXY" == "1" ]]; then
  PROXY_UNIT="Environment=SPARKBOARD_PROXY=1
Environment=SPARKBOARD_PROXY_TRANSPARENT=$PROXY_TRANSPARENT
Environment=SPARKBOARD_VLLM_UPSTREAM=$VLLM_UPSTREAM${CLASSIFY_UPSTREAM:+
Environment=SPARKBOARD_CLASSIFY_UPSTREAM=$CLASSIFY_UPSTREAM}${CLASSIFY_MODEL:+
Environment=SPARKBOARD_CLASSIFY_MODEL=$CLASSIFY_MODEL}
Environment=SPARKBOARD_CLASSIFY_SAMPLE=$CLASSIFY_SAMPLE${VLLM_API_KEY:+
Environment=SPARKBOARD_VLLM_API_KEY=$VLLM_API_KEY}${INJECT_AUTH:+
Environment=SPARKBOARD_INJECT_AUTH=$INJECT_AUTH}
Environment=SPARKBOARD_CLASSIFY_CONCURRENCY=$CLASSIFY_CONCURRENCY"
fi

cat > "/etc/systemd/system/${SVC_NAME}.service" <<UNIT
[Unit]
Description=Sparkboard -- GPU and system telemetry
Documentation=file://$APP_DIR/README.md
After=network.target

[Service]
Type=simple
User=$SVC_USER
Group=$SVC_USER
WorkingDirectory=$APP_DIR
Environment=SPARKBOARD_DB=$DATA_DIR/metrics.db
Environment=SPARKBOARD_BIND=$BIND
Environment=SPARKBOARD_PORT=$PORT
Environment=SPARKBOARD_INTERVAL=$INTERVAL
$PROXY_UNIT
ExecStart=$APP_DIR/venv/bin/python -m app.server
Restart=always
RestartSec=3

$DOCKER_UNIT

# Matching a listening socket to the process that owns it means reading
# /proc/<pid>/fd for processes owned by other users. Without these two the
# ports still appear, but with no owner attached.
# These survive NoNewPrivileges: ambient capabilities are inherited across
# execve rather than gained at it, which is what they were added for.
AmbientCapabilities=CAP_SYS_PTRACE CAP_DAC_READ_SEARCH
CapabilityBoundingSet=CAP_SYS_PTRACE CAP_DAC_READ_SEARCH

# Hardening. Two exceptions, both load-bearing:
#   PrivateDevices stays off -- nvidia-smi needs /dev/nvidia*.
#   ProtectProc stays at default -- hiding other users' /proc entries would
#   defeat the capabilities above.
NoNewPrivileges=yes
ProtectProc=default
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=$DATA_DIR
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
RestrictNamespaces=yes
LockPersonality=yes

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --quiet "$SVC_NAME"
systemctl restart "$SVC_NAME"
ok "service enabled and started"

# ---------------------------------------------------------------- verify

step "Verifying"

HEALTHY=0
for _ in $(seq 1 20); do
  if curl -fsS "http://${BIND}:${PORT}/api/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  sleep 0.5
done

if [[ $HEALTHY -eq 1 ]]; then
  ok "responding on http://${BIND}:${PORT}"
  SUMMARY=$(curl -fsS "http://${BIND}:${PORT}/api/info" 2>/dev/null | \
    python3 -c 'import json,sys
d=json.load(sys.stdin)
print(f"{d.get(\"hostname\")} | {d.get(\"gpu_name\") or \"no GPU\"} | driver {d.get(\"driver_version\") or \"n/a\"} | unified memory: {d.get(\"unified_memory\")}")' 2>/dev/null || true)
  [[ -n "$SUMMARY" ]] && ok "$SUMMARY"
else
  warn "no response yet -- check: journalctl -u $SVC_NAME -n 40 --no-pager"
fi

# ------------------------------------------------------------------ nginx

install -d -m 755 /etc/nginx/snippets 2>/dev/null || true
cat > "$NGINX_SNIPPET" <<NGINX
# Sparkboard -- add this inside the same server{} block that serves Open WebUI:
#   include $NGINX_SNIPPET;
# then: sudo nginx -t && sudo systemctl reload nginx

location = /gpu { return 301 /gpu/; }

location /gpu/ {
    proxy_pass http://${BIND}:${PORT}/;

    proxy_http_version 1.1;
    proxy_set_header Host              \$host;
    proxy_set_header X-Real-IP         \$remote_addr;
    proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto \$scheme;

    # The live view is server-sent events. Without these nginx buffers the
    # stream and the dashboard sits there looking frozen.
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 24h;
    chunked_transfer_encoding off;
}
NGINX
ok "nginx snippet written to $NGINX_SNIPPET"

say ""
step "Done"
say ""
say "  ${bold}One line left to add.${off} Open your nginx site config"
say "  ${dim}(the one with your Open WebUI proxy_pass, often /etc/nginx/sites-enabled/default)${off}"
say "  and put this inside the same ${bold}server { }${off} block:"
say ""
say "      ${grn}include $NGINX_SNIPPET;${off}"
say ""
say "  Then:"
say "      sudo nginx -t && sudo systemctl reload nginx"
say ""
say "  The dashboard will be at ${bold}https://<your-host>/gpu/${off}"
say ""
if [[ "$PROXY" == "1" ]]; then
  say "  ${bold}Activity feed is on.${off} Point your vLLM clients (or Open WebUI)"
  say "  at the proxy instead of vLLM directly:"
  say ""
  if [[ "$PROXY_TRANSPARENT" == "1" ]]; then
    say "  ${bold}Transparent mode is on.${off} The proxy answers at /v1 exactly like"
    say "  vLLM, so clients pointed at this host:port need ${bold}no change${off}."
    say "  Existing clients keep using:"
    say ""
    say "      ${grn}http://<this-host>:$PORT/v1${off}"
    say ""
    say "  (This assumes vLLM has moved to $VLLM_UPSTREAM and the proxy has taken"
    say "  over its old port.)"
  else
    say "      ${grn}http://<this-host>:$PORT/vllm/v1${off}   ${dim}(was $VLLM_UPSTREAM/v1)${off}"
    say ""
    say "  Only traffic through that path gets summarized. The feed shows category"
    say "  labels only -- never prompt text. To expose it publicly, add a /vllm/"
    say "  location to nginx; otherwise it stays reachable only on the host."
  fi
  say ""
fi
say "  ${dim}logs     journalctl -u $SVC_NAME -f${off}"
say "  ${dim}restart  sudo systemctl restart $SVC_NAME${off}"
say "  ${dim}remove   sudo ./install.sh --uninstall${off}"
say ""
