#!/usr/bin/env bash
# One-time root setup for nightly updates. From the stack folder:
#   sudo bash ops/install-maintenance.sh
# Re-run it after editing nightly-update.sh (it copies the script into place).
# On plain Linux (not WSL), reboots are enabled straight away; on WSL they
# wait for ops/windows-setup.ps1.
set -euo pipefail
OPS=$(cd "$(dirname "$0")" && pwd)
STACK=$(dirname "$OPS")

echo "-> nightly script + 02:00 timer"
install -m 0755 -o root -g root "$OPS/nightly-update.sh" /usr/local/sbin/stack-nightly
mkdir -p /etc/stack-maintenance
printf 'STACK=%q\nOWNER=%q\n' "$STACK" "$(stat -c %U "$STACK")" > /etc/stack-maintenance/stack.env
grep -qi microsoft /proc/version || touch /etc/stack-maintenance/allow-reboot
cat > /etc/systemd/system/stack-nightly.service <<'EOF'
[Unit]
Description=Nightly OS + Docker stack update (02:00)
After=docker.service network-online.target
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/stack-nightly
TimeoutStartSec=3h
Nice=10
IOSchedulingClass=idle
EOF
cat > /etc/systemd/system/stack-nightly.timer <<'EOF'
[Unit]
Description=Run stack-nightly at 02:00 every day
[Timer]
OnCalendar=*-*-* 02:00:00
AccuracySec=1min
# Not Persistent: if the PC was off/asleep at 2am, don't update + reboot mid-day.
Persistent=false
[Install]
WantedBy=timers.target
EOF

echo "-> apt: fold the 06:00 unattended run into the 02:00 window"
systemctl disable --now apt-daily-upgrade.timer apt-daily.timer >/dev/null 2>&1 || true

echo "-> docker daemon: log rotation + live-restore"
DJ=/etc/docker/daemon.json
cp -n "$DJ" "$DJ.bak-$(date +%F)" 2>/dev/null || true
python3 - "$DJ" <<'EOF'
import json,sys,os
p=sys.argv[1]; d=json.load(open(p)) if os.path.exists(p) else {}
d.setdefault("log-driver","json-file")
d.setdefault("log-opts",{"max-size":"50m","max-file":"3"})
d["live-restore"]=True          # dockerd upgrades no longer restart every container
json.dump(d,open(p,"w"),indent=4)
EOF

echo "-> sysctl: keep the model's pages in RAM rather than swap"
cat > /etc/sysctl.d/90-ai-stack.conf <<'EOF'
vm.swappiness=10
fs.inotify.max_user_watches=524288
EOF
sysctl -q --system

systemctl daemon-reload
systemctl restart docker
systemctl enable --now stack-nightly.timer
# Recreate containers so the new log limits apply to them
echo "-> recreating containers (llama needs ~5 min to load the model)"
cd "$STACK" && docker compose up -d --force-recreate
echo
systemctl list-timers stack-nightly.timer --no-pager
[[ -f /etc/stack-maintenance/allow-reboot ]] && echo "Done." || echo "Done. On WSL, reboots stay off until ops/windows-setup.ps1 has been run in an admin PowerShell."
