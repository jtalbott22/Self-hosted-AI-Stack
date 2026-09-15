# Changelog

All notable changes to Sparkboard are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [1.1.0]

Runs containerised, and on a second machine: a Windows 11 box with a discrete
RTX 4070, under WSL2. The dashboard and every panel are unchanged — this
release is about what it takes to gather the same numbers from inside a
container on a platform that reports less than a bare-metal Linux host does.

### Added

- Container image and a compose service block, with the app served from
  `app.server:app` rather than a systemd unit.
- Docker Engine API backend over the Unix socket, used in preference to
  shelling out to `docker ps` / `docker stats`. A slim image has no docker
  binary; this also reads raw counters instead of parsing values back out of
  formatted strings like `1.04GiB`. The CLI path remains as a fallback.
- `HOST_PROC` support: with the host's `/proc` mounted, psutil reports the
  machine rather than the container — memory, uptime, per-core CPU, the
  process table and the socket table all follow.
- Storage panel, driven by `DISK_PATHS`. Entries may carry a display label
  (`/host/disks/c=C:`), and a configured-but-absent mount is shown as
  unreadable rather than silently dropped.
- `capabilities` in `/api/info`: whether the host is WSL, whether the GPU does
  per-process accounting, whether a CPU temperature sensor exists. The frontend
  uses these to distinguish "nothing is happening" from "this platform cannot
  say", which look identical in the data.
- Heaviest-processes fallback for the GPU process table, used where the driver
  reports no compute apps. The panel retitles itself and explains the
  substitution instead of showing an empty table.
- Both environment naming styles are accepted (`SPARKBOARD_INTERVAL` and
  `SAMPLE_INTERVAL`, and so on), so an existing unit or compose file keeps
  working.
- Retention is configurable (`RETENTION_DAYS`, `RAW_RETENTION_HOURS`).
- Service tags for llama.cpp, ComfyUI, LibreChat, Jupyter, SearXNG,
  MeiliSearch, MongoDB and Playwright.

### Changed

- Temperature and power thresholds scale to the part. Power is read against
  the card's own enforced cap where the driver reports one, and temperature
  bands shift up for a discrete card — a 4070 at 65 °C is working, not in
  trouble, and colouring it red would teach the reader to ignore the colour.
- `nvidia-smi` discovery checks `/usr/lib/wsl/lib` and honours an explicit
  `NVIDIA_SMI_PATH`, so the driver can be reached with no container toolkit.
- The listening-ports panel says when it is looking at a container's own
  namespace rather than the host's, and its capability hint now covers the
  container case as well as the systemd one.
- httpx and httpcore are pinned to WARNING. The container inventory makes a
  request per container per sweep, which at INFO buried the service's own log.

### Fixed

- The unified-memory heuristic no longer fires on size alone. A discrete card
  whose VRAM happens to match the visible system RAM — entirely possible under
  WSL2, where the ceiling is whatever `.wslconfig` says — was being drawn as a
  shared pool. The size coincidence now only counts when the part's name also
  says it draws from system memory.
- The classifier survives an upstream that rejects `chat_template_kwargs`, as
  some llama.cpp builds do. The field is sent once, and on a 400 it is dropped
  for the life of the process instead of failing every label.

## [1.0.0]

First public release.

### Monitoring

- Live telemetry over Server-Sent Events, sampled from `nvidia-smi` and psutil.
- Historical windows (5m / 1h / 6h / 24h / 7d / 30d) backed by a two-tier SQLite
  store (raw samples plus minute rollups), with on-read bucketing and retention.
- Long windows draw each bucket's peak behind its mean so short spikes survive
  averaging.
- Prometheus exposition at `/metrics`.

### Built for unified memory (GB10 / Grace-Blackwell)

- Detects shared CPU/GPU DRAM and shows a single pool instead of two
  double-counted gauges; falls back to independent scales on discrete cards.
- Reconstructs GPU memory from the per-process view when `nvidia-smi` reports
  the framebuffer fields as `[N/A]`.
- Never fabricates a value — absent sensors (fan, power cap, framebuffer size)
  render as "—", not `0`.

### Containers and ports

- Per-container CPU, memory, published ports, and network I/O, with a trend
  sparkline that follows the time slicer.
- Listening-port-to-process attribution, tagged by service and cross-referenced
  against container port publications.
- GPU process list (what's holding VRAM).

### Optional LLM activity feed

- A reverse proxy in front of vLLM that forwards requests transparently
  (streaming and non-streaming) and asynchronously classifies each prompt into
  one of a fixed set of categories.
- **Privacy by construction:** prompt text is never written or logged; only a
  category label and token count are kept. The category set is closed, so a
  label cannot become a paraphrase of a prompt.
- Live scrolling feed with adjustable speed and a category rollup.
- **Transparent mode** (`SPARKBOARD_PROXY_TRANSPARENT`) mounts the proxy at
  `/v1` as a drop-in for vLLM, so existing clients are intercepted with no
  config change.
- **Dedicated classifier** support (`SPARKBOARD_CLASSIFY_UPSTREAM` /
  `SPARKBOARD_CLASSIFY_MODEL`) so labelling can run on a small model on its own
  port instead of competing with your main model.
- **API-key aware:** forwards the client's `Authorization` header, and
  authenticates its own classification calls when vLLM requires a key
  (`SPARKBOARD_VLLM_API_KEY`); optional `SPARKBOARD_INJECT_AUTH` lets the proxy
  hold the key so clients don't have to.
- Handles reasoning models (e.g. Qwen3.x) that return `content: null` while
  thinking — disables the thinking phase for label calls and parses reasoning
  fields defensively.

### Interface

- CPU load shown as Idle / Low / Medium / High tiers with an expandable
  per-core view.
- Threshold colours (green / amber / red) on temperature, utilization, power,
  and CPU, chosen to stay distinguishable for red-green colour vision.
- Light and dark themes (follows OS preference by default).
- Inline tooltips explaining abbreviations; keyboard- and touch-accessible.
- Mobile layout: single-column, vertical-scroll only, with wide tables
  scrolling inside their own cards.

### Install

- `install.sh` creates a service user, a virtualenv under `/opt/sparkboard`, a
  systemd unit, and an nginx snippet. Re-running upgrades in place and keeps
  history.
