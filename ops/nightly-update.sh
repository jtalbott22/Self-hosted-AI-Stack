#!/usr/bin/env bash
# Nightly maintenance for the AI stack. Runs as root at 02:00 from
# stack-nightly.timer (installed by ops/install-maintenance.sh, which also
# writes /etc/stack-maintenance/stack.env with STACK and OWNER).
#
#   1. Back up Open WebUI's Postgres DB and Home Assistant's /config (7 days kept)
#   2. Upgrade Ubuntu packages (all origins, keeps existing config files)
#   3. Pull newer images, rebuild local images, recreate changed containers
#   4. Clean up superseded images
#   5. Reboot if a package asked for it (only between 01:00-05:59). On WSL,
#      only once ops/windows-setup.ps1 has added the task that starts WSL again.
#
# Run by hand:  sudo /usr/local/sbin/stack-nightly        (logs: ops/logs/)
# Dry-ish run:  sudo NO_REBOOT=1 /usr/local/sbin/stack-nightly

set -uo pipefail
[[ -f /etc/stack-maintenance/stack.env ]] && . /etc/stack-maintenance/stack.env
: "${STACK:?STACK not set (run ops/install-maintenance.sh)}"
OWNER=${OWNER:-$(stat -c %U "$STACK")}
LOG_DIR=$STACK/ops/logs
BK=$STACK/ops/backups/nightly
STAMP=$(date +%F)
mkdir -p "$LOG_DIR" "$BK"
LOG=$LOG_DIR/nightly-$STAMP.log
exec > >(tee -a "$LOG") 2>&1
export DEBIAN_FRONTEND=noninteractive
FAILED=()
step() { echo; echo "== $(date '+%F %T')  $*"; }
try()  { "$@" || { echo "!! failed: $*"; FAILED+=("$*"); }; }

cd "$STACK" || exit 1

step "Backups"
docker ps --format '{{.Names}}' | grep -qx postgres &&
  try bash -c "docker exec postgres pg_dump -U openwebui -Fc openwebui > '$BK/openwebui-$STAMP.dump'"
[[ -d "$STACK/homeassistant" ]] && try tar czf "$BK/homeassistant-$STAMP.tgz" -C "$STACK" homeassistant
for f in docker-compose.yml docker-compose.yaml compose.yaml; do [[ -f $f ]] && cp -a "$f" "$BK/$f-$STAMP"; done
find "$BK" -type f -mtime +7 -delete

step "Ubuntu packages"
try apt-get update -q
try apt-get -y -q -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold full-upgrade
try apt-get -y -q autoremove --purge
command -v snap >/dev/null && try snap refresh

step "Docker images"
try docker compose pull --ignore-buildable --quiet
try docker compose build --pull --quiet

# Playwright must match the playwright package inside Open WebUI.
PV=$(docker run --rm --entrypoint pip ghcr.io/open-webui/open-webui:main show playwright 2>/dev/null | awk '/^Version:/{print $2}')
if [[ -n "$PV" ]]; then
  CUR=$(grep -oP '^PLAYWRIGHT_VERSION=\K.*' .env || true)
  if [[ "$PV" != "$CUR" ]]; then
    echo "Playwright ${CUR:-unset} -> $PV"
    if grep -q '^PLAYWRIGHT_VERSION=' .env; then sed -i "s/^PLAYWRIGHT_VERSION=.*/PLAYWRIGHT_VERSION=$PV/" .env
    else echo "PLAYWRIGHT_VERSION=$PV" >> .env; fi
    chown "$OWNER": .env; chmod 600 .env
    try docker compose pull --quiet playwright
  fi
fi

try docker compose up -d
step "Cleanup"
docker image prune -f
docker builder prune -f --filter until=168h >/dev/null

step "Status"
docker ps --format '{{.Names}}\t{{.Status}}' | sort
UNHEALTHY=$(docker ps --filter health=unhealthy --format '{{.Names}}')
[[ -n "$UNHEALTHY" ]] && FAILED+=("unhealthy: $UNHEALTHY")
chown -R "$OWNER": "$LOG_DIR" "$BK"

{ echo "last run: $(date '+%F %T')"; echo "failures: ${#FAILED[@]}"; printf '  %s\n' "${FAILED[@]}"; } > "$LOG_DIR/last-run.txt"
find "$LOG_DIR" -name 'nightly-*.log' -mtime +30 -delete

if [[ -f /var/run/reboot-required ]]; then
  H=$(date +%-H)
  if [[ -n "${NO_REBOOT:-}" ]]; then echo "Reboot required (skipped: NO_REBOOT)"
  elif [[ ! -f /etc/stack-maintenance/allow-reboot ]]; then echo "Reboot required, but reboots aren't enabled (WSL: run ops/windows-setup.ps1; Linux: touch /etc/stack-maintenance/allow-reboot)"
  elif (( H < 1 || H > 5 )); then echo "Reboot required, deferred to tonight (it's ${H}:00)"
  else
    step "Rebooting: $(cat /var/run/reboot-required.pkgs 2>/dev/null | tr '\n' ' ')"
    sync; systemctl reboot
  fi
fi
exit 0
