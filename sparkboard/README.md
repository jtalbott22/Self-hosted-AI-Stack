<div align="center">

# Sparkboard

**A lightweight GPU and system dashboard for NVIDIA GPUs, including DGX Spark / GB10 and other unified-memory systems.**

Live telemetry, historical trends, Docker stats, listening-port attribution, and an optional privacy-preserving view of what your local LLM is doing — served through the nginx you already have in front of Open WebUI.

No build step · no CDN · no external services · one Python process · one SQLite file

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey.svg)
![No build step](https://img.shields.io/badge/frontend-no%20build%20step-orange.svg)

[Changelog](CHANGELOG.md)

</div>

---

> **Screenshots:** Add dashboard screenshots to `docs/` when publishing a
> release. Keeping screenshots in the repository makes the project easier to
> understand before installation.

---

## Why this exists

The GB10 in a DGX Spark is a great little inference box, but it's an awkward one to watch. `nvidia-smi` reports several fields as `[N/A]` because the part has no discrete framebuffer, off-the-shelf dashboards assume a conventional GPU and render those gaps as zeros, and standing up Prometheus + Grafana + dcgm-exporter + node-exporter is four services and a provisioning story for a single machine you already know how to reach.

Sparkboard is one service that samples `nvidia-smi` and psutil on a timer, stores to SQLite, and serves both an API and a self-contained dashboard. It understands the Spark's quirks: unified memory is shown as a single shared pool, missing sensors read as "—" rather than 0, and GPU memory that `nvidia-smi` won't report is reconstructed from the per-process view.

It still exposes a Prometheus endpoint, so if you later want Grafana, that's a scrape target rather than a rewrite.

## Features

- **Live + historical** — real-time stream over Server-Sent Events, with 5m / 1h / 6h / 24h / 7d / 30d windows backed by automatic minute-rollups. Long windows draw each bucket's peak behind its mean so short spikes don't average away.
- **Built for unified memory** — detects GB10-style shared DRAM and shows one pool for CPU + GPU instead of two double-counted gauges. Falls back to independent scales on discrete cards automatically.
- **Honest about missing sensors** — fan speed, power cap, and framebuffer size are `[N/A]` on GB10; they render as "—", never as a fabricated zero.
- **CPU load tiers** — cores grouped into Idle / Low / Medium / High as a stacked bar you can read at a glance, with an expandable per-core view underneath. A single-threaded pin looks different from a balanced load.
- **Threshold colours** — temperature, utilization, power, and CPU readouts tint green / amber / red by band, with thresholds chosen to stay distinguishable for red-green colour vision.
- **Containers** — every running container with CPU, memory, published ports, and network I/O, plus a per-row trend sparkline that follows the time slicer.
- **Listening ports** — every TCP listener matched to the process that owns it, tagged (vLLM, uvicorn, nginx, Ollama, Open WebUI…), and cross-referenced against container port publications.
- **GPU processes** — what's actually holding VRAM, which is usually the question you have when a model won't load.
- **Optional LLM activity feed** — a live, scrolling view of _what kind_ of work your vLLM server is doing, by category, **never showing prompt text**. See [Activity feed & privacy](#activity-feed--privacy).
- **Light and dark themes**, inline tooltips explaining every abbreviation, and a mobile-friendly layout (single-column, vertical-scroll only — wide tables scroll inside their own cards).
- **Prometheus endpoint** at `/metrics` for when you outgrow the built-in views.

## Running in a container

Sparkboard installs on bare metal as a systemd service (below), and also runs
as a container — which is how it reaches a Windows host through WSL2. The
container build needs three mounts to see the machine rather than itself: the
host's `/proc`, the Docker socket, and whichever filesystems you want in the
storage panel. `docker-compose.sparkboard.yml` is a working service block and
[INSTALL.md](INSTALL.md) walks through it, including what each access path
grants and which panels degrade without it.

## Requirements

- Linux (built and tested on Ubuntu 24.04, ARM64/GB10; works on x86 too)
- Python 3.10+
- `nvidia-smi` on `PATH` for GPU metrics (the service still runs and shows host metrics without it)
- An existing nginx reverse proxy if you want to serve it at a nice URL (optional — you can also just hit the port)
- Docker in a `docker` group if you want container stats (optional)

## Quick start

### Bare metal Linux

```bash
git clone <repository-url>
cd sparkboard
sudo ./install.sh
```

The installer creates a dedicated service account, installs Sparkboard under
`/opt/sparkboard`, stores its SQLite database under `/var/lib/sparkboard`, and
registers a systemd service. The default bind address is `127.0.0.1`, so the
dashboard is not exposed to the LAN unless you explicitly change it.

### Docker Compose

For a containerized deployment, use the included
`docker-compose.sparkboard.yml` as a service block. The container needs access
to host `/proc` and, if desired, the Docker socket and selected filesystems.
These mounts are powerful and should be treated as part of Sparkboard's
security boundary.

The installer creates a `sparkboard` system user, builds a virtualenv under `/opt/sparkboard`, installs and starts a systemd unit, and writes an nginx snippet to `/etc/nginx/snippets/sparkboard.conf`.

To serve it through your existing nginx, add one line inside the same `server { }` block that already proxies Open WebUI (often `/etc/nginx/sites-enabled/default`):

```nginx
include /etc/nginx/snippets/sparkboard.conf;
```

```bash
sudo nginx -t && sudo systemctl reload nginx
```

The dashboard is then at **`https://<your-host>/gpu/`**.

Re-running `install.sh` upgrades in place and keeps your history. Prefer not to use the installer? It's a standard FastAPI app — `pip install -r requirements.txt` and run `python -m app.server` behind any reverse proxy.

## Activity feed & privacy

The optional activity feed answers a question people reasonably ask about a shared LLM box — _"what is this thing doing?"_ — without exposing what anyone actually typed.

**How it works.** With the feed enabled, Sparkboard runs a thin OpenAI-compatible reverse proxy in front of vLLM. Requests pass through untouched (streaming and non-streaming), so from a client's perspective it's just vLLM on a different port. Separately, each prompt is sent back to vLLM with an instruction to label it with the closest of a **fixed set of ~14 categories** — "writing code", "debugging", "drafting an email", "translating", and so on. Only that label is kept and shown.

**The privacy model, precisely:**

- Prompt text is **never written to disk and never logged**.
- It exists in memory only for the moment it takes to classify, inside one async task, and is dropped when that task returns.
- What leaves the proxy is a **category label and a token count** — never the text that produced them.
- The category list is **closed**. The model picks the nearest label; it is never asked to describe freely, so a label cannot become a paraphrase of a prompt.
- Classification is **fire-and-forget**. If it fails, is slow, or is switched off, real traffic is unaffected — the proxy's one job is to forward.

This is a structural guarantee, not a filter applied after the fact. The test suite asserts it directly: prompts containing distinctive words are pushed through, and the feed output is scanned to confirm none of those words appear.

**A note on "public trust."** If your goal is to reassure people, showing that the system is busy and varied — a live mix of categories — does that. Publishing verbatim prompts does the opposite once someone realises their query landed on a dashboard. This feature is deliberately built for the former.

**The categories.** The classifier picks the single closest label from this fixed list (or `other`): writing code, debugging code, reviewing code, writing or editing text, drafting an email or message, writing a story or creative piece, summarizing or extracting, translating, answering a question, explaining a concept, planning or organizing, data analysis or math, role-play or conversation. The set is closed — the model can't invent labels or describe freely, which is what keeps a label from becoming a paraphrase of a prompt. To customize, edit `CATEGORIES` in `app/proxy.py`.

**Enabling it:**

```bash
SPARKBOARD_PROXY=1 SPARKBOARD_VLLM_UPSTREAM=http://127.0.0.1:8000 sudo ./install.sh
```

Then point your clients (or Open WebUI's OpenAI endpoint) at `http://<host>:9101/vllm/v1` instead of vLLM directly. Only traffic through the proxy is summarized. By default the proxy is reachable only on the host; there's a commented `/vllm/` block in the nginx snippet if you want to expose it.

### Intercepting more than one client

If several clients hit vLLM (chat UIs, agents, a teammate's editor plugin), you don't want to repoint each one. Two approaches:

**Explicit (default).** Enable the proxy, and point clients at `…:9101/vllm/v1`. Each client you repoint gets summarized; the rest bypass it. Fine for one or two clients you control.

**Transparent (drop-in).** Move vLLM to a private port, put the proxy on vLLM's *old* port, and enable transparent mode so the proxy answers at `/v1` exactly like vLLM. Every existing client keeps its current config unchanged and is intercepted automatically — including remote ones you can't easily edit.

```bash
# vLLM moved to 8100; proxy takes over 8000, transparent, reachable on the network
SPARKBOARD_PROXY=1 SPARKBOARD_PROXY_TRANSPARENT=1 \
  SPARKBOARD_PORT=8000 SPARKBOARD_BIND=0.0.0.0 \
  SPARKBOARD_VLLM_UPSTREAM=http://127.0.0.1:8100 \
  sudo ./install.sh
```

With transparent mode, bind vLLM's host port to `127.0.0.1` so it's reachable *only* through the proxy, not directly. `SPARKBOARD_BIND=0.0.0.0` is needed only if off-box clients hit the proxy.

### Using a dedicated classifier model (recommended for busy servers)

By default the label call reuses your main model. On a loaded server — or with a large reasoning model, which spends tokens thinking before it answers — that adds avoidable work to the same GPU slots serving real traffic.

Instead, run a small model on its own port and point the classifier at it. Labeling then never competes with inference:

```bash
# main vLLM on :8000, a tiny classifier on :8001
SPARKBOARD_PROXY=1 \
  SPARKBOARD_VLLM_UPSTREAM=http://127.0.0.1:8000 \
  SPARKBOARD_CLASSIFY_UPSTREAM=http://127.0.0.1:8001 \
  SPARKBOARD_CLASSIFY_MODEL=<the-small-model's-served-name> \
  sudo ./install.sh
```

A 0.5B–3B instruct model is plenty for picking one label from a fixed list — `qwen2.5:0.5b`, `llama3.2:1b`, or a small `ministral`/`gemma` served through a second vLLM. Non-reasoning models are ideal here: no thinking phase, near-instant, tiny memory footprint.

`SPARKBOARD_CLASSIFY_MODEL` is **required** when the classify upstream differs — the small instance doesn't serve your big model's name. Sparkboard logs a warning at startup if you forget it.

The feed is **off by default** — nothing about the proxy runs unless you opt in.

## Securing vLLM with an API key

If vLLM is reachable on the network, lock it down with a key. Start vLLM with `--api-key <key>` (or `VLLM_API_KEY=<key>`), and it will reject any request without `Authorization: Bearer <key>`.

The proxy sits in the middle, so two request paths need that key:

- **Passthrough** (client → proxy → vLLM): the proxy forwards the client's `Authorization` header unchanged, so each client just uses the key as if talking to vLLM directly. No proxy setting needed.
- **Classification** (proxy → vLLM): the label call is a request the proxy makes itself, so it needs its own copy of the key. Set `SPARKBOARD_VLLM_API_KEY`:

```bash
SPARKBOARD_PROXY=1 \
  SPARKBOARD_VLLM_UPSTREAM=http://127.0.0.1:8000 \
  SPARKBOARD_VLLM_API_KEY=<the-same-key-vllm-uses> \
  sudo ./install.sh
```

Without this, passthrough still works but classification fails with 401 (`classified` stays 0, `classify_errors` climbs).

**Optional — proxy holds the key so clients don't.** With `SPARKBOARD_INJECT_AUTH=1`, the proxy adds the vLLM key to any forwarded request that arrives without one. Clients then reach the proxy unauthenticated while vLLM stays locked. A client that sends its own key keeps it (the proxy never overwrites), so both styles coexist. Only do this if the proxy itself is access-controlled (e.g. behind nginx with its own auth, or on a trusted network) — otherwise you've just moved the open door.

> **Note:** environment values in a systemd unit are readable via `systemctl cat sparkboard` by any local user. For a shared host, consider an `EnvironmentFile=` with `chmod 600` instead of inline `Environment=` lines, and restart the service after editing.

## Configuration

Environment variables, set in `/etc/systemd/system/sparkboard.service` (then `sudo systemctl daemon-reload && sudo systemctl restart sparkboard`):

| Variable | Default | Notes |
|---|---|---|
| `SPARKBOARD_BIND` | `127.0.0.1` | Bind address; keep on loopback and let nginx be the front door |
| `SPARKBOARD_PORT` | `9101` | |
| `SPARKBOARD_INTERVAL` | `2` | Seconds between telemetry samples |
| `SPARKBOARD_DB` | `/var/lib/sparkboard/metrics.db` | SQLite path |
| `SPARKBOARD_SERVICES_INTERVAL` | `10` | Seconds between container/port scans |
| `SPARKBOARD_DOCKER` | `1` | Set `0` at install time to skip container stats |
| `SPARKBOARD_LOGLEVEL` | `INFO` | |
| **Activity feed** | | |
| `SPARKBOARD_PROXY` | `0` | `1` enables the vLLM proxy and feed |
| `SPARKBOARD_VLLM_UPSTREAM` | `http://127.0.0.1:8000` | Where real vLLM listens |
| `SPARKBOARD_CLASSIFY_UPSTREAM` | _(main upstream)_ | Separate endpoint for label calls — point at a small model |
| `SPARKBOARD_CLASSIFY_MODEL` | _(request's model)_ | Model for labelling; **required** if the classify upstream differs |
| `SPARKBOARD_VLLM_API_KEY` | _(none)_ | Key the proxy uses for its classification calls when vLLM requires auth |
| `SPARKBOARD_INJECT_AUTH` | `0` | `1` = proxy supplies the vLLM key for forwarded requests missing one |
| `SPARKBOARD_CLASSIFY` | `1` | `0` = pure transparent proxy, no labelling |
| `SPARKBOARD_CLASSIFY_SAMPLE` | `1.0` | Fraction of requests to classify (0–1) |
| `SPARKBOARD_CLASSIFY_CONCURRENCY` | `2` | Max concurrent classification calls |
| `SPARKBOARD_CLASSIFY_API_KEY` | _(vLLM key)_ | Override key for a separate classify upstream |
| `SPARKBOARD_CLASSIFY_MAXCHARS` | `2000` | Longest prompt slice sent to the classifier |
| `SPARKBOARD_CLASSIFY_TIMEOUT` | `20` | Timeout (seconds) for the classify call |
| `SPARKBOARD_CLASSIFY_MODEL` | _(request's model)_ | Override the model used for labelling |

**Storage.** Raw samples are kept ~26 hours, minute-rollups ~120 days; at the default interval this settles around 60–80 MB for host + GPU and stops growing. Container history adds roughly 1 MB per container per month.

**Privileges.** Port-to-process attribution needs `CAP_SYS_PTRACE` and `CAP_DAC_READ_SEARCH` (the unit grants these — narrow, read-only, and they survive `NoNewPrivileges`). Container stats need the service user in the `docker` group, which is effectively root on the host; the installer says so when it makes that change, and `SPARKBOARD_DOCKER=0` skips it.

## API

Everything is relative to the mount point, so these work at `/gpu/…` behind nginx or at the root.

| Endpoint | Returns |
|---|---|
| `GET /api/info` | Host and GPU facts (queried once at startup) |
| `GET /api/now` | Latest full sample |
| `GET /api/live` | In-memory ring (~5 minutes) |
| `GET /api/history?range=1h&gpu=0` | Bucketed series, ~700 points |
| `GET /api/containers/history?range=1h` | Per-container series on a shared timeline |
| `GET /api/services` | Containers and listening ports, current state |
| `GET /api/prompts/feed?after=<seq>` | Activity labels + category rollup (no prompt text) |
| `GET /api/stream` | SSE, one message per sample |
| `GET /api/health` | Liveness, sample/error counts, store size, proxy stats |
| `GET /metrics` | Prometheus text exposition |
| `ANY /vllm/<path>` | Transparent vLLM proxy (only when the feed is enabled) |

## Architecture

```
                          ┌───────────────────────────────┐
   browser ── /gpu/ ────► │  nginx (your existing proxy)  │
                          └───────────────┬───────────────┘
                                          │  proxy_pass :9101
                          ┌───────────────▼───────────────┐
                          │        Sparkboard (FastAPI)    │
                          │                                │
   sampler thread ───────►│  collector ─┐                 │
   (nvidia-smi, psutil)   │             ├─► SQLite (2-tier)│
   services thread ──────►│  services ──┘   raw + rollups  │
   (docker, ports)        │                                │
                          │  SSE stream ─► live dashboard   │
                          └───────────────┬────────────────┘
                                          │  (feed enabled)
                          ┌───────────────▼────────────────┐
   LLM clients ──/vllm/──►│  classifying proxy ──► vLLM     │
                          │  label only ─► activity feed    │
                          └────────────────────────────────┘
```

- `app/collector.py` — `nvidia-smi` + psutil sampling, defensive about `[N/A]`, unified-memory detection and reconstruction.
- `app/store.py` — two-tier SQLite time-series (raw + minute rollups) with retention and on-read bucketing.
- `app/services.py` — Docker inventory and listening-port attribution.
- `app/proxy.py` — the classifying vLLM reverse proxy (transparent forward + async labelling).
- `app/server.py` — FastAPI wiring, SSE, and the sampler threads.
- `static/` — the dashboard: one HTML file, a dependency-free canvas charting engine, and the app controller.

Adding a metric is a column in `store.py`, a field in `collector.py`, and an entry in the `PANELS` array in `static/app.js`.

## Troubleshooting

<details>
<summary><b>Fan and power-cap show as “—”</b></summary>

Expected on GB10. The module is passively managed and doesn't expose those sensors, so the collector reports them as absent rather than as zero. A dash means "no reading"; a zero would be a claim.
</details>

<details>
<summary><b>GPU memory is “—” but processes clearly hold VRAM</b></summary>

`nvidia-smi --query-gpu=memory.used,memory.total` returns `[N/A]` on GB10 — there's no discrete framebuffer for those fields. Sparkboard reconstructs GPU allocation by summing the per-process query and takes the pool size from host RAM, labelling both as derived. History collected before this reconstruction stays empty for those buckets and fills in going forward.
</details>

<details>
<summary><b>A chart says “not reported for this range”</b></summary>

That sensor returned nothing for every bucket in the window — distinct from a flat line at zero, which is a real reading of zero.
</details>

<details>
<summary><b>Page loads but charts stay frozen / a table stops updating</b></summary>

nginx is buffering the event stream. Confirm `proxy_buffering off;` is in the `/gpu/` block and reload nginx. All `/api/` responses are also sent `Cache-Control: no-store`; the containers header shows how long ago its data arrived, which is the quickest way to confirm freshness.
</details>

<details>
<summary><b>404 on <code>/gpu/app.js</code></b></summary>

The trailing slash on `proxy_pass http://127.0.0.1:9101/;` is what strips the `/gpu/` prefix. Without it nginx forwards the path verbatim and the app has no such route.
</details>

<details>
<summary><b>Activity feed is empty</b></summary>

Check the proxy is enabled (`systemctl show sparkboard -p Environment | tr ' ' '\n' | grep PROXY`) and that clients are actually hitting `/vllm/…` rather than vLLM directly. Only traffic through the proxy is summarized. The feed also starts empty and fills as requests arrive.
</details>

<details>
<summary><b>Ports show but the process column says “not attributed”</b></summary>

The unit didn't get its capabilities. Check `systemctl show sparkboard -p AmbientCapabilities`; if empty, re-run the installer. Published container ports are held by `docker-proxy`, so those rows name the container instead.
</details>

<details>
<summary><b>On mobile it zooms out and pans around instead of scrolling vertically</b></summary>

That happens when a page element is wider than the screen. Sparkboard pins the page to viewport width and keeps wide tables scrolling *inside their own cards*, so the page itself only scrolls vertically. If you still see it, hard-refresh to clear a cached stylesheet. Pinch-zoom still works — it's only the involuntary zoom-out that's prevented.
</details>

<details>
<summary><b>Service won't start</b></summary>

```bash
journalctl -u sparkboard -n 50 --no-pager
curl -s localhost:9101/api/health | python3 -m json.tool   # bypass nginx
```
</details>

## Uninstall

```bash
sudo ./install.sh --uninstall   # keeps the database
sudo ./install.sh --purge       # removes everything
```

Then drop the `include` line from your nginx config and reload.

## Contributing

Issues and pull requests welcome. The project is deliberately small and dependency-light — the frontend has no build step (the charting is a hand-rolled canvas engine), and the backend is plain FastAPI + psutil + `nvidia-smi`. Please keep that spirit: no frontend framework, no CDN, no new heavy runtime dependencies without a good reason.

A few conventions:

- The one rule that matters most: **never fabricate a value.** A missing sensor renders as "—", never as a fabricated `0`. This is the whole reason the dashboard is trustworthy on hardware with `[N/A]` sensors; please preserve it.
- Adding a metric is roughly: a column in `store.py`, a field in `collector.py`, and an entry in the `PANELS` (or `READOUTS`) array in `static/app.js`.
- The activity-feed classifier keeps **labels only, never prompt text** — that's a structural guarantee, not a filter. Don't add anything that logs or persists prompt content.

## Security

- The dashboard binds to `127.0.0.1` by default; put it behind your own reverse proxy rather than exposing it directly. `SPARKBOARD_BIND=0.0.0.0` opens it to the network — only do that behind a firewall/VPN.
- If you run the vLLM activity feed and expose the proxy, secure vLLM with an API key (see [Securing vLLM with an API key](#securing-vllm-with-an-api-key)).
- Found a security issue? Please open an issue marked security, or contact the maintainer directly rather than filing full exploit details publicly.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built for the NVIDIA DGX Spark (GB10 Grace Blackwell). The dashboard is deliberately dependency-free on the frontend: the charting is a small hand-rolled canvas engine, so there's no build step and it runs on an isolated network.
