# Self-Hosted Local AI Stack

A practical, local-first AI workstation built around **Open WebUI + llama.cpp**, with web search, RAG, Jupyter code execution, image generation, text-to-speech, and an optional system-monitoring dashboard.

The goal of this repository is not to provide a generic cloud AI server. It is a reproducible example of how to assemble a capable private AI environment on a single NVIDIA-equipped workstation.

## What it provides

| Component | Purpose |
|---|---|
| **llama.cpp** | Runs a local GGUF language model with an OpenAI-compatible API |
| **Open WebUI** | Main chat interface and orchestration layer |
| **PostgreSQL + pgvector** | Persistent application database and vector storage for RAG |
| **SearXNG** | Self-hosted metasearch backend for web search |
| **Playwright** | Loads JavaScript-heavy pages for the web loader |
| **Apache Tika** | Extracts text from PDFs, Office files, and other documents |
| **JupyterLab** | Code execution and notebook-based analysis |
| **ComfyUI** | Local image generation |
| **Kokoro** | Local text-to-speech |
| **Open Terminal** | Gives Open WebUI a shell and filesystem interface |
| **Sparkboard** *(optional)* | Local system, storage, container, and AI activity monitoring |

Everything runs through Docker Compose and communicates over the private Compose network unless a port is intentionally published.

## Architecture

```text
                         ┌──────────────────────┐
                         │      Open WebUI      │
                         │    localhost:3000    │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼────────────────────────┐
             │                      │                        │
             ▼                      ▼                        ▼
        ┌──────────┐          ┌──────────┐            ┌──────────┐
        │ llama.cpp│          │  Jupyter │            │ ComfyUI  │
        │  :8000   │          │  :8888   │            │  :8188   │
        └────┬─────┘          └──────────┘            └──────────┘
             │
             ▼
       Local GGUF model

  Open WebUI also connects to:
    PostgreSQL/pgvector → RAG
    Tika                  → document extraction
    SearXNG + Playwright  → web search
    Kokoro                → TTS
    Open Terminal         → shell/filesystem
    Sparkboard            → optional local monitoring
```

## Hardware expectations

This configuration is tuned around a workstation-class NVIDIA GPU and a model that can use system RAM for MoE experts.

### Reasonable starting point

- NVIDIA GPU with **12 GB+ VRAM**
- **64 GB+ system RAM** recommended for the included 35B MoE configuration
- SSD/NVMe storage
- Linux or Windows 11 + WSL2
- NVIDIA Container Toolkit
- Docker Engine + Docker Compose

The included llama.cpp command uses:

```text
-ngl 99
--n-cpu-moe 999
```

That intentionally keeps GPU-resident work on the GPU while allowing MoE expert weights to live in system RAM. This makes a substantially larger model practical on a GPU with limited VRAM, at the cost of additional system-memory use and lower performance than a model that fits entirely in VRAM.

If your machine has less RAM or VRAM, use a smaller GGUF and adjust the llama.cpp command accordingly.

## Prerequisites

### Linux

Install:

1. NVIDIA driver
2. NVIDIA Container Toolkit
3. Docker Engine
4. Docker Compose

Verify GPU access:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### Windows + WSL2

Use:

- Windows 11
- WSL2
- Ubuntu
- NVIDIA driver installed on Windows
- NVIDIA Container Toolkit inside WSL
- Docker Engine inside WSL

Docker Desktop is not required for this configuration.

## Install

Clone the repository:

```bash
git clone <repository-url>
cd Self-hosted-AI-Stack
```

Create the directories used by the Compose file:

```bash
mkdir -p models comfy comfy-models notebooks searxng
```

The Compose file deliberately uses **relative paths** rather than a machine-specific `/home/...` path. This keeps the repository portable between machines and users.

### Download a model

The included configuration expects:

```text
models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
models/mmproj-F16.gguf
```

For example:

```bash
pip install -U "huggingface_hub[cli]"

hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  mmproj-F16.gguf \
  --local-dir ./models
```

You can use another GGUF model, but update the `-m` argument in `docker-compose.yaml`.

### Configure secrets

Copy the example:

```bash
cp .env.example .env
```

Generate strong random values:

```bash
openssl rand -hex 32
```

Use separate values for:

- `POSTGRES_PASSWORD`
- `WEBUI_SECRET_KEY`
- `JUPYTER_TOKEN`
- `OPEN_TERMINAL_API_KEY`

Never commit `.env`.

### Configure SearXNG

Create:

```text
searxng/settings.yml
```

with:

```yaml
use_default_settings: true

server:
  secret_key: "replace-with-a-random-secret"
  limiter: false

search:
  formats:
    - html
    - json
```

The `json` format is required for Open WebUI's SearXNG integration.

### Start the stack

```bash
docker compose up -d
docker compose ps
```

Watch the model server:

```bash
docker compose logs -f llama
```

The first startup can take several minutes while images and the model are loaded.

## Open WebUI

Open:

```text
http://localhost:3000
```

The first account created becomes the administrator.

The model should appear as:

```text
qwen
```

If it does not, check:

```bash
docker compose logs llama
```

## Open Terminal

Open Terminal is intentionally **not published to the host network**.

In Open WebUI, configure the Open Terminal integration using:

```text
URL:      http://open-terminal:8000
API Key:  value of OPEN_TERMINAL_API_KEY
Auth:     Bearer
```

This component is powerful: it provides a shell/filesystem interface and should be treated as a trusted-user capability.

## Web search

Open WebUI is configured to use:

```text
SearXNG → Playwright
```

The important internal endpoints are:

```text
http://searxng:8080
ws://playwright:3000
```

If web search returns nothing, verify that SearXNG has JSON enabled.

## RAG and documents

The stack uses:

- Apache Tika for document extraction
- `BAAI/bge-m3` for embeddings
- `BAAI/bge-reranker-v2-m3` for reranking
- PostgreSQL/pgvector for vector storage

Embedding and reranking models are downloaded when first needed.

## Image generation

Open WebUI connects to ComfyUI at:

```text
http://comfyui:8188
```

Put compatible checkpoints under:

```text
comfy-models/checkpoints/
```

ComfyUI itself is available at:

```text
http://localhost:8188
```

## Jupyter / code execution

Jupyter is available directly at:

```text
http://localhost:8888
```

Open WebUI is configured to use the internal endpoint:

```text
http://jupyter:8888
```

The notebook directory is:

```text
./notebooks
```

## Sparkboard

Sparkboard is an optional monitoring component included as a local project under:

```text
./sparkboard
```

It can monitor:

- CPU and memory
- storage
- running containers
- service state
- historical metrics
- optional AI activity classification

It listens on:

```text
http://localhost:9102
```

Sparkboard uses host `/proc` and the Docker socket for monitoring. Docker socket access is security-sensitive even when mounted read-only, so do not expose Sparkboard directly to an untrusted network.

### WSL storage monitoring

The default Compose configuration includes:

```text
/mnt/c → C:
/mnt/d → D:
```

These mounts are useful under WSL but should be removed or changed for a native Linux installation.

The `DISK_PATHS` environment variable controls the labels shown by Sparkboard.

## Ports

| Port | Service | Purpose |
|---:|---|---|
| 3000 | Open WebUI | Main interface |
| 8000 | llama.cpp | OpenAI-compatible model API |
| 8080 | SearXNG | Search interface |
| 8188 | ComfyUI | Image generation UI |
| 8880 | Kokoro | TTS API |
| 8888 | Jupyter | Notebook/code environment |
| 9102 | Sparkboard | Monitoring dashboard |

Playwright, Tika, PostgreSQL, and Open Terminal do not publish host ports.

## Security

This is a **single-user / trusted-user stack**, not a hardened multi-tenant platform.

In particular, Open Terminal can provide extremely powerful host/container access. Do not expose it directly to the Internet.

Before exposing Open WebUI externally:

- use HTTPS
- use strong authentication
- keep secrets out of Git
- consider binding administrative services to localhost
- understand what host directories are mounted into containers

## Updating

Pull new images:

```bash
docker compose pull
docker compose up -d
```

Your persistent application data is stored in named Docker volumes.

Back up important data before making major image or database-version changes.

## Repository layout

```text
.
├── docker-compose.yaml
├── .env.example
├── .gitignore
├── README.md
├── models/
├── comfy/
├── comfy-models/
├── notebooks/
├── searxng/
└── sparkboard/
```

Model weights, generated data, secrets, databases, and other machine-specific state should not be committed to Git.

## License

Choose a license appropriate for the code you publish. The repository itself is primarily configuration and integration glue around the respective upstream projects; each upstream component remains subject to its own license and terms.
