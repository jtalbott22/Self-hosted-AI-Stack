# Local AI Stack — Open WebUI on a Gaming PC

A full-featured, self-hosted AI workstation running entirely on one Windows desktop.
No API keys, no subscriptions, nothing leaves the machine.

Chat, web search, RAG over your own documents, code execution, image generation,
text-to-speech, and a real Linux terminal the model can drive — all behind a single
`docker compose up -d`.

---

## What's in the box

| Service | Image | What it does |
|---|---|---|
| **llama.cpp** | `ghcr.io/ggml-org/llama.cpp:server-cuda` | Serves the LLM over an OpenAI-compatible API |
| **Open WebUI** | `ghcr.io/open-webui/open-webui:main` | The front end everything else plugs into |
| **PostgreSQL + pgvector** | `pgvector/pgvector:pg17` | App database and vector store for RAG |
| **SearXNG** | `searxng/searxng:latest` | Private metasearch — the web search backend |
| **Playwright** | `mcr.microsoft.com/playwright` | Renders JS-heavy pages so search results are actually readable |
| **Apache Tika** | `apache/tika:latest-full` | Text extraction from PDFs, Office docs, etc. |
| **JupyterLab** | `quay.io/jupyter/scipy-notebook` | Sandboxed code execution / code interpreter |
| **Open Terminal** | `ghcr.io/open-webui/open-terminal` | Gives the model a shell, a filesystem, and a file browser |
| **ComfyUI** | `yanwk/comfyui-boot:cu128-slim` | Image generation |
| **Kokoro** | `ghcr.io/remsky/kokoro-fastapi-cpu` | Text-to-speech (CPU, so it doesn't fight for VRAM) |

---

## System Requirements

```
+--------------------------------------------------------------+
|                                                              |
|                      MINIMUM  SYSTEM                         |
|                                                              |
|   CPU .......... 6-core x86-64 (AMD Ryzen 5 / Intel i5)      |
|   RAM .......... 32 GB                                       |
|   GPU .......... NVIDIA, 8 GB VRAM, driver 550+             |
|   DISK ......... 120 GB free, SSD                            |
|   OS ........... Windows 11 + WSL2, or Linux                 |
|   NET .......... Broadband (first pull is ~40 GB)            |
|                                                              |
+--------------------------------------------------------------+
+--------------------------------------------------------------+
|                                                              |
|                    RECOMMENDED  SYSTEM                       |
|                                                              |
|   CPU .......... 8P/16E cores (i7-13700K class or better)   |
|   RAM .......... 64-96 GB                                    |
|   GPU .......... NVIDIA, 12-16 GB VRAM                       |
|   DISK ......... 250 GB free NVMe + bulk HDD for models      |
|                                                              |
+--------------------------------------------------------------+
```

**Reference build** (what this config is tuned for): Intel i7-13700KF, 96 GB DDR5,
RTX 4070 12 GB, NVMe + HDD, Windows 11 Pro, Docker Engine inside WSL2 Ubuntu 24.04.

### Why those numbers

**RAM matters more than VRAM here.** The model is a Mixture-of-Experts, and the config
pushes every expert tensor to system RAM (`--n-cpu-moe 999`) while keeping attention
layers on the GPU. That's what lets a 35B model run on a 12 GB card. The tradeoff is
that the weights live in your RAM instead — budget ~24 GB for the model, plus ~8 GB for
the rest of the containers. On 32 GB you'll want a smaller model (14B or an 8B).

**VRAM is shared.** The GPU is running the LLM's attention layers, ComfyUI's diffusion
model, and — if you have no integrated graphics — your actual desktop. ComfyUI is
launched with `--reserve-vram 0.6` so it yields some headroom. Expect to close one
before hammering the other on 8 GB.

**Disk adds up fast.** A Q4 GGUF of a 35B model is ~20 GB. Container images are ~25 GB.
The embedding and reranker models pull another ~4 GB on first use. An SDXL checkpoint is
~7 GB. Postgres, chat history, and uploaded documents grow from there.

### Prerequisites

- NVIDIA driver installed **on Windows** (not inside WSL)
- WSL2 with Ubuntu 24.04
- Docker Engine installed inside WSL — **not** Docker Desktop
- NVIDIA Container Toolkit installed inside WSL

Verify GPU passthrough before you start:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

If you don't see your card, stop and fix that first — nothing else will work.

---

## Install

### 1. Create the host directories

```bash
mkdir -p models comfy comfy-models notebooks searxng
```

### 2. Download a model

```bash
pip install -U "huggingface_hub[cli]"
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --local-dir ./models
```

Swap in whatever GGUF you like — just update the `-m` path in `docker-compose.yml`.
If you're on less RAM, drop to a smaller model and remove `--n-cpu-moe 999` so it runs
fully on the GPU.

### 3. Write your `.env`

Copy `.env.example` to `.env` and fill in every value:

```env
POSTGRES_PASSWORD=
WEBUI_SECRET_KEY=
JUPYTER_TOKEN=
OPEN_TERMINAL_API_KEY=
```

Generate a value for each:

```bash
openssl rand -hex 32
```

```bash
chmod 600 .env
```

Do not commit this file. Add it to `.gitignore`.

### 4. Configure SearXNG

This one bites everybody. SearXNG returns HTML only by default, and Open WebUI needs
JSON. Create `searxng/settings.yml`:

```yaml
use_default_settings: true
server:
  secret_key: "paste-another-openssl-rand-hex-32-here"
  limiter: false
search:
  formats:
    - html
    - json
```

Without the `json` format, web search silently returns nothing.

### 5. Bring it up

```bash
docker compose up -d
docker compose ps
```

First start pulls ~25 GB of images and loads the model. Give it several minutes —
the llama healthcheck has a 300-second grace period for exactly this reason.

```bash
docker compose logs -f llama
```

Wait for the server to report it's listening before you try to chat.

---

## Post-install: the Open WebUI side

Most of the stack is wired through environment variables, so it configures itself.
These are the parts that still need a human.

### Step 1 — Create your admin account

Open `http://localhost:3000`. **The first account created becomes the administrator.**
If you're exposing this machine at all, register immediately, before anyone else can.

### Step 2 — Confirm the model is there

The model dropdown should already list `qwen`. If it's empty, Open WebUI can't reach
llama.cpp — check `docker compose logs llama`.

### Step 3 — Connect Open Terminal

This is the only connection that can't be done purely from env vars, and the settings
menu is genuinely confusing here: **Integrations appears twice.** Once under *Personal →
Services*, once under *Admin → Tools*. You want the Admin one. The personal entry binds
the terminal to your account only and pushes the API key into your browser.

1. Click your name (bottom left) → **Settings**
2. Under the **Admin** section, click **Integrations**
3. Scroll to the **Open Terminal** section — it has its own section, do not add it under
   *External Tool Servers*, or you lose the file browser and the terminal sidebar
4. Click **+** and fill in:
   - **URL:** `http://open-terminal:8000`
   - **API Key:** your `OPEN_TERMINAL_API_KEY` from `.env`
   - **Auth Type:** Bearer
   - **Chat Uploads:** Default
5. **Save**, then refresh the page

To use it: in the chat input, click the terminal (cloud) icon and pick your terminal
under **System**. Ask it "what operating system are you running on?" — it should run a
command and answer.

### Step 4 — Check function calling is Native

Terminal use is the most demanding thing you can ask a model to do. Go to **Workspace →
Models**, edit your model, and confirm **Function Calling** is set to **Native**, not
Legacy. Native is the default as of v0.10.0. Legacy falls back to prompt-based tool
calling and often won't fire the terminal at all.

### Step 5 — Verify web search

**Admin Settings → Web Search.** Engine should read `searxng`, Web Loader Engine should
read `playwright`, and the **Playwright WebSocket URL** field should show
`ws://playwright:3000`. If that field is blank, type it in and hit Save — it's a
persisted setting, so once it exists in the database the environment variable stops
seeding it and the UI becomes the source of truth.

Then test it: toggle **Web Search** on in a chat and ask about something recent.

### Step 6 — Image generation

**Admin Settings → Images.** Engine is `comfyui` and the URL is `http://comfyui:8188`.
You still need a checkpoint: drop an SDXL or Flux safetensors into `comfy-models/checkpoints/`,
then pick it in the Model field. ComfyUI's own interface is at `http://localhost:8188`
if you want to build a custom workflow and export the API JSON.

### Step 7 — Audio

**Admin Settings → Audio.** TTS should already point at Kokoro. Hit the play button on
any assistant message to test. Voices are selectable in the same panel — `af_sky` is the
default here.

### Step 8 — Documents and RAG

**Admin Settings → Documents.** Content extraction is Tika; embedding is `bge-m3` with
`bge-reranker-v2-m3` for hybrid search. Both models download on first use, so the very
first document you upload will take a minute. Upload a PDF to **Workspace → Knowledge**
and ask a question about it.

---

## Ports

| Port | Service | Binding |
|---|---|---|
| 3000 | Open WebUI | All interfaces |
| 8188 | ComfyUI | All interfaces |
| 8000 | llama.cpp | localhost only |
| 8080 | SearXNG | localhost only |
| 8880 | Kokoro | localhost only |
| 8888 | JupyterLab | localhost only |

Playwright, Tika, Postgres, and Open Terminal publish **nothing**. They're reachable only
over the internal Compose network, by service name. Keep it that way.

---

## Security

**Open Terminal is a root shell on your machine.** Anyone who can reach it can pull and
run arbitrary containers, mount host directories, use host networking, and manage every
container on the host. There is no isolation boundary between users. This is a
single-trusted-user setup — treat it as such:

- Use a real random `OPEN_TERMINAL_API_KEY`, never a placeholder
- Never publish port 8000 for it
- If you put Open WebUI behind a reverse proxy on a public hostname, put authentication in
  front of it and set `WEBUI_URL` to the public URL

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Terminal connection times out | `localhost` in the URL. Inside a container that's the container. Use `http://open-terminal:8000` |
| Web search returns nothing | SearXNG `settings.yml` missing the `json` format |
| Playwright errors on every page | Image tag doesn't match the version pinned in the Open WebUI image |
| Model missing from dropdown | llama.cpp still loading, or OOM — check `docker compose logs llama` |
| Everything is glacially slow | Model doesn't fit; too many layers spilling to CPU. Use a smaller quant |
| `nvidia-smi` fails in a container | NVIDIA Container Toolkit not installed in WSL |

Two commands that answer most questions:

```bash
docker exec openwebui curl -s http://open-terminal:8000/health
docker exec openwebui pip show playwright | grep Version
```

The first should print `{"status": "ok"}`. The second must match the Playwright image
tag in `docker-compose.yml` — if the Open WebUI image bumps its pinned version, bump
both the tag and the `playwright@` in that service's command.

---

## Updating

```bash
docker compose pull
docker compose up -d
```

Your data lives in named volumes (`openwebui-data`, `pgdata`, `open-terminal`) and
survives container replacement. Back up `pgdata` before a major version jump.
