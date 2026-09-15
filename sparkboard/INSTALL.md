# Sparkboard Installation Guide

This is the same Sparkboard that runs on the GB10, packaged to run as one
container inside the existing WSL2 Docker stack. Same dashboard, same panels,
same activity feed — containers, listening ports, GPU and host telemetry, live
view over SSE, and history from 5m out to 30d in SQLite.

Nothing to install on Windows. No Prometheus, no Grafana, no agent.

---

## Install

**1. Replace the old `sparkboard/` folder next to the compose file.**

```
your-stack/
├── docker-compose.yml
├── comfy/
├── notebooks/
└── sparkboard/          <- this folder
    ├── Dockerfile
    ├── app/             <- the app is a package now, not one file
    ├── static/
    ├── docker-compose.sparkboard.yml
    └── requirements.txt
```

If you still have the previous single-file build, delete `app.py` from that
folder. It is not used and will only confuse the next person to look.

**2. Merge the service block.** Copy the `sparkboard:` service from
`docker-compose.sparkboard.yml` into your `docker-compose.yml`, and the
`sparkboard-data:` entry into its top-level `volumes:`. Or leave the file where
it is and pass both to compose:

```bash
cd ~/stack
docker compose -f docker-compose.yml -f docker-compose.sparkboard.yml up -d --build sparkboard
```

The old service block will not work unchanged — this build needs the Docker
socket and a couple of capabilities it didn't ask for before. Use the new one.

**3. Build and start only this service.** Nothing else in the stack restarts:

```bash
docker compose up -d --build sparkboard
```

**4. Check what it can see.**

```bash
curl -s localhost:9102/api/health | python3 -m json.tool
```

Four things worth reading in that response:

| Field | Expected | If it's wrong |
|---|---|---|
| `nvidia_smi` | `/usr/bin/nvidia-smi` | GPU section below |
| `gpus` | `1` | GPU section below |
| `host_proc` | `/host/proc` | the `/proc` mount is missing |
| `capabilities.wsl` | `true` | you're not where you think you are |

Then confirm the two panels that need extra access:

```bash
curl -s localhost:9102/api/services | python3 -c \
  "import json,sys; d=json.load(sys.stdin); \
   print('docker:', d['docker']['available'], d['docker'].get('error') or ''); \
   print('containers:', len(d['docker']['containers'])); \
   print('ports:', len(d['listeners']['ports']), 'unattributed:', d['listeners']['unattributed'])"
```

You want `docker: True` with your real container count, and `unattributed: 0`.

**5. Open it.** <http://localhost:9102> — WSL2 forwards the port to Windows
automatically.

---

## Put it behind nginx

Paste `nginx-sparkboard-windows.conf` into the `server` block of the Windows
nginx config, beside the Open WebUI route, and `nginx -s reload`. It's then at
`https://<your-host>/sparkboard/`.

The trailing slash matters — the page loads its API and assets by relative
path, which is what lets it sit under any prefix without a rebuild. The
redirect on the first line of that file covers the bare URL.

Sparkboard has no authentication of its own, and the ports table is a map of
everything listening on the box. If that hostname answers from outside your
network, put basic auth or an IP restriction on the location block; both are
written out at the bottom of that file.

---

## Host access and permissions

Four separate access paths, and each panel degrades on its own if one is
missing. Nothing here is all-or-nothing.

**The GPU** — `runtime: nvidia` with `NVIDIA_DRIVER_CAPABILITIES=utility`, the
same NVIDIA Container Toolkit path `llama` and `comfyui` already use.
`utility` injects `nvidia-smi` and NVML and nothing else, so Sparkboard reads
the card without pulling in the CUDA runtime and without reserving any VRAM. It
will not compete with llama.cpp or ComfyUI for the 4070's 12 GB.

If the toolkit is ever broken or removed, there's a fallback needing no toolkit
at all — borrow the driver Windows already exposes to WSL. Replace the
`runtime` and `NVIDIA_*` lines with:

```yaml
    environment:
      - NVIDIA_SMI_PATH=/usr/lib/wsl/lib/nvidia-smi
      - LD_LIBRARY_PATH=/usr/lib/wsl/lib
    volumes:
      - /usr/lib/wsl:/usr/lib/wsl:ro
```

**The host** — `/proc:/host/proc:ro` plus `HOST_PROC=/host/proc`. This is the
one that matters most and it is easy to miss why. Without it, psutil reports
the *container*: two processes, 200 MB of memory, one loopback socket. Every
number would be real and every one would be about the wrong machine. With it,
memory, uptime, per-core CPU, the process table and the socket table all come
from the WSL VM.

**Docker** — the Engine API over `/var/run/docker.sock`. This build talks to
the socket directly rather than shelling out to `docker`, because a slim
container image has no docker binary in it. Read the note in the compose file
about what mounting that socket grants before you decide you're happy with it;
`:ro` is less protective than it looks.

**Port attribution** — `pid: host` and `cap_add: [SYS_PTRACE]`. Matching a
listening port to the process holding it means reading `/proc/<pid>/fd` for
processes owned by other users. Without these the ports still all appear, with
the owner column reading "not attributed" and a note on the panel saying so.

---

## The activity feed

Off by default. When on, Sparkboard sits in front of llama.cpp as a reverse
proxy: every request is forwarded untouched, and separately the prompt is sent
back to the model to be labelled with one of fourteen fixed categories.
**Only the label is kept.** Prompt text is never written to disk, never
logged, and lives in memory only for the moment it takes to classify.

To turn it on, uncomment the proxy block in the compose file and point Open
WebUI at Sparkboard instead of at llama directly:

```
OPENAI_API_BASE_URL=http://sparkboard:9102/vllm/v1
```

and set the upstream to wherever llama actually binds. In this stack that is
port 8000 — `--port 8000` in llama's command line — not llama.cpp's 8080
default, and 8080 on this host is searxng. Pointed at the wrong one the feed
simply stays empty while the log fills with connection errors:

```yaml
- SPARKBOARD_PROXY=1
- SPARKBOARD_VLLM_UPSTREAM=http://llama:8000
```

Two settings matter more here than they do on the GB10. That box has headroom
to spare; this one is serving a 35B MoE on a 12 GB card with experts spilled to
system RAM, and every label is a small generation of its own competing for the
same slot:

```yaml
- SPARKBOARD_CLASSIFY_SAMPLE=0.5        # label half the requests, not all
- SPARKBOARD_CLASSIFY_CONCURRENCY=1     # never more than one label in flight
```

Turn the sample rate down further if you notice first-token latency moving. The
feed is a nice thing to watch; it is not worth slowing down the thing it is
watching.

One compatibility note, already handled: some llama.cpp builds reject the
`chat_template_kwargs` field that suppresses Qwen's thinking phase. Sparkboard
sends it once, and if the server refuses, drops it permanently and carries on.
You'll see one line in the log the first time and nothing after.

---

## Reading the numbers honestly

Several things are measured from inside WSL2 rather than from Windows, and the
dashboard now says so on the face of it rather than leaving you to work it out.

**Per-process VRAM is unavailable.** The WSL driver doesn't do compute-process
accounting, so `nvidia-smi` reports total VRAM in use but not who is holding
it. The panel detects this and switches to the heaviest processes by resident
memory, retitling itself "Heaviest processes" and saying why underneath. The
GPU memory *total* is accurate — that's the number to watch when ComfyUI and
llama.cpp are both resident against `--reserve-vram 0.6`.

**Memory shows about 48 GB, not 96.** WSL2 claims roughly half of system RAM by
default and that ceiling is what the VM sees. Since `--n-cpu-moe 999` pushes
MoE experts into system RAM, more headroom is worth having anyway. Create
`C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=72GB
processors=20
```

Then `wsl --shutdown` from PowerShell and bring the stack back up.

**Listening ports are the VM's, not Windows'.** You'll see everything inside
WSL2, which includes every published container port — that's where Docker's
proxy processes live. You will *not* see Windows-side listeners: the nginx on
443, RDP, anything native. On the GB10 that distinction doesn't exist; here it
does.

**Heaviest processes lists Linux processes only.** llama.cpp, ComfyUI's python,
the Open WebUI node process, Docker itself. Windows applications aren't visible
from inside the VM.

**CPU temperature is blank.** No hwmon sensors inside WSL2. The dashboard shows
an em-dash rather than a zero. GPU temperature comes from the driver and is
real.

**Storage shows what you mount.** A WSL2 deployment can mount `/mnt/c` and `/mnt/d`
and labels them C: and D:, so the storage panel covers the 1.8 TB SSD and the
5.5 TB HDD alongside the WSL virtual disk. Drop the mounts and the panel just
shows the VM's own disk.

**Temperature and power bands are scaled to the card.** A 4070 at 65 °C and
180 W is a card doing its job. The thresholds that suit a fanless 140 W GB10
module would paint that red, so on a discrete card the bands shift up and power
is read against the card's own enforced cap.

---

## Troubleshooting

**`"docker": false` with a permissions error**

The socket is mounted but the container can't read it. Check the socket's group
on the WSL side (`ls -l /var/run/docker.sock`); if the daemon runs rootless the
path is different — `$XDG_RUNTIME_DIR/docker.sock` — and needs mounting from
there instead, with `DOCKER_SOCK` set to match.

**Ports listed but `unattributed` is non-zero**

`pid: host` or `cap_add: [SYS_PTRACE]` missing from the service. The panel says
this itself, at the bottom of the table.

**`nvidia_smi: not found`**

Check the toolkit works for a bare container:

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=utility \
  python:3.12-slim nvidia-smi -L
```

If that fails but `llama` and `comfyui` still run, re-register the runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

If it still fails, switch to the `/usr/lib/wsl` fallback above.

**Memory reads ~200 MB and only one port is listed**

The `/proc` mount didn't take. `curl -s localhost:9102/api/health` will show
`"host_proc": null`. Sparkboard is reporting on itself.

**Charts empty on anything but LIVE**

Normal at first — history accumulates from when the container starts. The 7d
and 30d views fill in over time. The footer shows the row counts climbing.

**Page loads but every value is an em-dash**

The collector thread threw. `docker compose logs sparkboard` prints why. An
em-dash is deliberate everywhere: it means "no reading", never zero.

---

## Knobs

Both naming styles work; the `SPARKBOARD_`-prefixed one wins if you set both.

| Variable | Default | What it does |
|---|---|---|
| `HOST_LABEL` | host `/proc` hostname | Name in the header |
| `SAMPLE_INTERVAL` | `2` | Seconds between telemetry samples |
| `SERVICES_INTERVAL` | `10` | Seconds between container/port sweeps |
| `RETENTION_DAYS` | `120` | How long minute rollups are kept |
| `RAW_RETENTION_HOURS` | `26` | How long raw samples are kept |
| `DISK_PATHS` | `/` | Comma-separated; `path=Label` to rename |
| `HOST_PROC` | `/host/proc` | Where the host's `/proc` is mounted |
| `DOCKER_SOCK` | `/var/run/docker.sock` | Engine socket for the containers panel |
| `SPARKBOARD_DOCKER` | `1` | Set `0` to drop the containers panel entirely |
| `NVIDIA_SMI_PATH` | auto | Explicit path to `nvidia-smi` |
| `SPARKBOARD_PROXY` | `0` | Activity feed / classifying proxy |
| `SPARKBOARD_VLLM_UPSTREAM` | `http://127.0.0.1:8000` | Inference server to front |
| `SPARKBOARD_CLASSIFY_SAMPLE` | `1.0` | Fraction of requests labelled |
| `SPARKBOARD_CLASSIFY_CONCURRENCY` | `2` | Ceiling on in-flight labels |

## Endpoints

- `GET /` — the dashboard
- `GET /api/info` — host and GPU facts, capabilities, filesystems
- `GET /api/now` — most recent full sample
- `GET /api/live` — in-memory ring, so a fresh tab has populated charts
- `GET /api/history?range=1h&gpu=0` — bucketed series; `5m` `1h` `6h` `24h` `7d` `30d`
- `GET /api/services` — containers and listening ports
- `GET /api/containers/history?range=1h` — per-container series
- `GET /api/prompts/feed` — activity labels and category rollup
- `GET /api/stream` — SSE, one message per sample
- `GET /api/health` — liveness, store stats, capabilities, proxy stats
- `GET /metrics` — Prometheus exposition

History returns roughly 700 points regardless of window, so a 30-day pull costs
about what a five-minute one does.
