# Self-Hosted Local AI Stack

A practical, local-first AI workstation built around **Open WebUI + llama.cpp**, with web search, RAG, Jupyter code execution, image generation, text-to-speech, and an optional system-monitoring dashboard.

The goal of this repository is not to provide a generic cloud AI server. It is a reproducible example of how to assemble a capable private AI environment on a single NVIDIA-equipped workstation.

<a href="docs/screenshots/chat-home.png"><img src="docs/screenshots/chat-home.png" alt="Open WebUI chat interface" width="280"></a>

## What it provides

| Component | Purpose |
|---|---|
| **llama.cpp** | Runs a local GGUF language model with an OpenAI-compatible API |
| **Inkling 2 Mini** *(optional)* | A second local model. Sparkboard's [model director](#model-director) shares the GPU between it and Qwen |
| **Open WebUI** | Main chat interface and orchestration layer |
| **PostgreSQL + pgvector** | Persistent application database and vector storage for RAG |
| **SearXNG** | Self-hosted metasearch backend for web search |
| **Playwright** | Loads JavaScript-heavy pages for the web loader |
| **Apache Tika** | Extracts text from PDFs, Office files, and other documents |
| **JupyterLab** | Code execution and notebook-based analysis |
| **ComfyUI** | Local image generation |
| **Kokoro** | Local text-to-speech |
| **Open Terminal** | Gives Open WebUI a shell and filesystem interface |
| **GitLab MCP** | Lets the model work with issues, merge requests, repos, and pipelines on your GitLab |
| **Sparkboard** *(optional)* | Local system, storage, container, and AI activity monitoring, container start/stop/restart, and the model director |
| **Avatar** *(optional)* | Talking 3D head in the browser. Built on [LAM](https://github.com/aigc3d/LAM), it uses llama.cpp, Kokoro, ScribeCrate (speech-to-text) and LAM Audio2Expression for lip-sync |

Everything runs through Docker Compose and communicates over the private Compose network unless a port is intentionally published.

## Architecture

<a href="docs/diagrams/architecture-stack-v2.png"><img src="docs/diagrams/architecture-stack-v2.png" alt="Architecture of the whole stack: you reach the avatar through CloudFront and nginx with Google sign-in, or on the LAN; the avatar and Open WebUI share one local llama.cpp model, the voice services, and search, RAG and tool services, and the avatar controls Home Assistant."></a>

From outside, your domain goes through CloudFront to nginx on the Windows host ([config](nginx/windows/)), which asks for a Google sign-in (oauth2-proxy, with authgate banning brute-force attempts); on the LAN, nginx lets you straight in. Open WebUI and the avatar share one local llama.cpp model and the voice services. Open WebUI also uses Postgres/pgvector for RAG, Tika to read documents, SearXNG and Playwright for web search, Jupyter for code, ComfyUI for images, Open Terminal and the GitLab MCP server. Sparkboard (optional) monitors the machine.

<a href="docs/diagrams/architecture-avatar-v2.png"><img src="docs/diagrams/architecture-avatar-v2.png" alt="What the avatar page talks to: Kokoro, Speaches and the LAM lip-sync and maker services for voice and face; the music service; the shared llama.cpp model; the Lens bridge for Open WebUI tools; and, per Google account, saved chats and home control through Home Assistant."></a>

The avatar page reaches everything through relative paths on its own nginx (`:8702` on the LAN, `/avatar/` behind the public nginx), so the browser never makes a cross-origin call. Saved chats and home control are kept per Google account, from the sign-in.

<sub>Both diagrams were drawn to the rules of the stack's own diagram-design tool and rendered by its `render_diagram` function on this stack (an Open WebUI tool, run through the Lens bridge). Sources: [stack](docs/diagrams/architecture-stack.html), [avatar](docs/diagrams/architecture-avatar.html), and the [generator](docs/diagrams/generate.py) that lays them out on the tool's 4px grid.</sub>

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

These commands, run inside the WSL Ubuntu shell, install Docker Engine and the NVIDIA Container Toolkit:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Close and reopen the WSL shell so the `docker` group membership takes effect, then check GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

To have the stack come back after a reboot, enable systemd in WSL (`/etc/wsl.conf` needs `systemd=true` under `[boot]`) and start Docker at boot:

```bash
grep -A1 '\[boot\]' /etc/wsl.conf
sudo systemctl enable docker containerd
```

Every service in `docker-compose.yaml` uses `restart: unless-stopped`, so the containers start again with Docker.

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
- `SPARKBOARD_CONTROL_KEY` *(optional: turns on Sparkboard's start/stop/restart buttons)*

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

Open WebUI reaches llama.cpp as an OpenAI-compatible connection (**Admin Panel → Connections**):

```text
URL: http://llama:8000/v1
```

<a href="docs/screenshots/connections.png"><img src="docs/screenshots/connections.png" alt="Open WebUI Connections settings" width="280"></a>

No API key is required for this internal connection. The built-in OpenAI API toggle can stay on with no key set if you don't also want a real OpenAI account wired in.

The model should appear as:

```text
qwen
```

If it does not, check:

```bash
docker compose logs llama
```

Configured models show up under **Admin Panel → Models**:

<a href="docs/screenshots/models.png"><img src="docs/screenshots/models.png" alt="Models configured in Open WebUI" width="280"></a>

## Open Terminal

Open Terminal is intentionally **not published to the host network**.

In Open WebUI, configure the Open Terminal integration using:

```text
URL:      http://open-terminal:8000
API Key:  value of OPEN_TERMINAL_API_KEY
Auth:     Bearer
```

This component is powerful: it provides a shell/filesystem interface and should be treated as a trusted-user capability.

## GitLab (MCP)

Agent stacks usually connect to systems like GitLab through an **MCP server**: a small service that exposes the GitLab API as tools the model can call. Open WebUI (v0.6.31+) can talk to MCP servers natively over Streamable HTTP, so no extra proxy is needed.

The `gitlab-mcp` service runs the community [zereight/gitlab-mcp](https://github.com/zereight/gitlab-mcp) server. It works with self-managed GitLab and gives the model tools for projects, files, branches, commits, issues, merge requests (including diffs and review comments), and pipelines.

### 1. Create a GitLab token

Create a dedicated GitLab user (e.g. `ai-bot`) and add it as **Developer** only on the projects the assistant should touch. Then create a personal access token for it with the `api` scope (use `read_api` for read-only).

All chats that use the connection act as this one GitLab identity, so its project membership is the real permission boundary.

### 2. Point the service at your instance

Add to `.env`:

```bash
GITLAB_API_URL=https://gitlab.example.com/api/v4
# Optional:
# GITLAB_PERMISSION_MODE=modify     # readonly | modify | full
# GITLAB_TOOLSETS=core              # e.g. core,pipelines,wiki
```

`modify` (the default here) lets the model create branches, commit files, open issues and merge requests, and comment, but blocks delete, cancel, and unprotect tools. Use `readonly` if you only want questions answered.

If your GitLab uses an internal CA, uncomment the `NODE_EXTRA_CA_CERTS` line and the certificate volume in `docker-compose.yaml`.

```bash
docker compose up -d gitlab-mcp
docker compose logs -f gitlab-mcp
```

### 3. Add the connection in Open WebUI

**Admin Panel → Settings → External Tools** (called **Integrations** in newer versions) → **+**:

| Field | Value |
|---|---|
| Type | **MCP (Streamable HTTP)**, not OpenAPI |
| URL | `http://gitlab-mcp:3002/mcp` |
| Auth | Bearer |
| Key | the GitLab token from step 1 |

The token stays in Open WebUI's database, not in `.env`. The server forwards it to GitLab on every request.

### 4. Use it

Enable the GitLab tools on a model (**Workspace → Models → Tools**) or per chat from the **+** menu. Set **Function Calling** to **Native** in the model's advanced parameters; the multi-step tool loop depends on it.

Example prompts:

- "Summarize open merge requests in `group/project` that are waiting on review."
- "Read `src/config.py` on `main` in `group/project`, fix the timeout default, commit it to a new branch, and open a merge request."
- "Why did the latest pipeline on `main` fail? Pull the failing job log."
- "Triage issues labeled `bug` opened this week and comment with a likely cause."

### Notes for local models

Every enabled tool's schema is sent with each request. The `core` toolset is about 40 tools, which a 35B model handles, but adding many toolsets fills context and makes tool selection less reliable. Add toolsets (`pipelines`, `wiki`, `milestones`, …) only as you need them.

The MCP server works through the GitLab API, which suits issues, MRs, reviews, pipelines, and small file edits. For changes that need a real checkout (multi-file refactors, running tests before pushing), have the model use **Open Terminal** with `git` and GitLab's [`glab`](https://gitlab.com/gitlab-org/cli) CLI instead: clone, edit, test, push, then `glab mr create`.

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

### Admin panel reference values

**Admin Panel → Web Search**

<a href="docs/screenshots/web-search-general.png"><img src="docs/screenshots/web-search-general.png" alt="Web Search settings" width="280"></a>
<a href="docs/screenshots/web-search-loader.png"><img src="docs/screenshots/web-search-loader.png" alt="Web Search loader settings" width="280"></a>

| Setting | Value |
|---|---|
| Web Search | On |
| Web Search Confirmation | Off |
| Web Search Engine | `searxng` |
| Searxng Query URL | `http://searxng:8080/search?q=<query>&format=json` |
| Search Result Count | 15 |
| Concurrent Requests | 5 |
| Web Loader Engine | `playwright` |
| Playwright WebSocket URL | `ws://playwright:3000` |
| Playwright Timeout | 10000 ms |

## RAG and documents

The stack uses:

- Apache Tika for document extraction
- `BAAI/bge-m3` for embeddings
- `BAAI/bge-reranker-v2-m3` for reranking
- PostgreSQL/pgvector for vector storage

Embedding and reranking models are downloaded when first needed.

### Admin panel reference values

**Admin Panel → Documents**

<a href="docs/screenshots/documents-embedding-rag.png"><img src="docs/screenshots/documents-embedding-rag.png" alt="Documents embedding and retrieval settings" width="280"></a>
<a href="docs/screenshots/documents-extraction.png"><img src="docs/screenshots/documents-extraction.png" alt="Documents content extraction settings" width="280"></a>

| Setting | Value |
|---|---|
| Content Extraction Engine | Tika |
| Tika Server URL | `http://tika:9998` |
| Tika Server Version | Tika 3.x |
| Text Splitter | Default (Character) |
| Markdown Header Text Splitter | On |
| Chunk Size | 1200 |
| Embedding Batch Size | 1 |
| Hybrid Search | On |
| Enrich Hybrid Search Text | Off |
| Full Context Mode | Off |
| Reranking Engine | Default (SentenceTransformers) |

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

### Admin panel reference values

**Admin Panel → Images → Create Image**

<a href="docs/screenshots/image-generation.png"><img src="docs/screenshots/image-generation.png" alt="Image generation settings" width="280"></a>

| Setting | Value |
|---|---|
| Image Generation Engine | ComfyUI |
| Base URL | `http://comfyui:8188` |
| Image Size | 1024x1024 |
| Steps | 25 |
| Image Prompt Generation | On |

### Image editing

Open WebUI can also route image *edits* through a separate ComfyUI workflow (**Admin Panel → Images → Edit Image**), pointed at the same `http://comfyui:8188` base URL. After uploading your workflow as API-format JSON under **ComfyUI Workflow**, map the workflow's node IDs to the fields Open WebUI needs to fill in, e.g.:

<a href="docs/screenshots/image-editing.png"><img src="docs/screenshots/image-editing.png" alt="Image editing workflow settings" width="280"></a>

| Field | Workflow input | Node ID |
|---|---|---|
| Image | `image` | 78 |
| Prompt | `prompt` | 435 |
| Model | `unet_name` | set to your checkpoint/unet loader node |

Node IDs are specific to your exported workflow graph and will differ from these.

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

Open WebUI has two places that point at Jupyter and both need the same URL and token: **Admin Panel → Code Execution** (used by the Code Interpreter tool mid-chat) and its **Code Interpreter Engine**, set to `jupyter (Legacy)`.

### Admin panel reference values

**Admin Panel → Code Execution**

<a href="docs/screenshots/code-execution.png"><img src="docs/screenshots/code-execution.png" alt="Code execution and interpreter settings" width="280"></a>

| Setting | Value |
|---|---|
| Jupyter URL | `http://jupyter:8888` |
| Jupyter Auth | Token |
| Jupyter Token | value of `JUPYTER_TOKEN` |
| Code Execution Timeout | 60s |
| Enable Code Interpreter | On |
| Code Interpreter Engine | jupyter (Legacy) |

## Text-to-speech

Open WebUI talks to Kokoro through the OpenAI-compatible TTS API:

<a href="docs/screenshots/text-to-speech.png"><img src="docs/screenshots/text-to-speech.png" alt="Audio / text-to-speech settings" width="280"></a>

| Setting | Value |
|---|---|
| Text-to-Speech Engine | OpenAI |
| API Base URL | `http://kokoro:8880/v1` |
| TTS Voice | `af_sky` |
| TTS Model | `kokoro` |
| Response Splitting | Punctuation |

Response Splitting controls how message text is chunked before being sent to TTS; Punctuation splits on sentence boundaries rather than sending one long block or splitting by paragraph.

## Diagram Design

`Tools/diagram_design.py` is an Open WebUI tool that adapts the [diagram-design](https://github.com/cathrynlavery/diagram-design) agent skill (MIT, by Cathryn Lavery). The skill draws 45 diagram types, including architecture, flowchart, sequence, state machine, ER, timeline, quadrant, Sankey, and Gantt, as one self-contained HTML + SVG file in a consistent editorial style. Diagrams render inline in the chat.

Open WebUI's own Skills feature takes a single markdown file, and this skill relies on about 60 reference docs and a library of example files, so it is packaged as a tool instead. It gives the model three functions:

| Function | What it does |
|---|---|
| `diagram_guide` | The workflow, the list of 45 types with when to use each, the design rules, and the style tokens |
| `diagram_reference` | The layout rules for one type plus a finished example file to adapt (`light`, `dark`, or `full` variant) |
| `render_diagram` | Runs the skill's self-check, saves the `.html` file, and renders it inline in the chat |

In the chat, each diagram fits the panel width and zooms like an image. It stays vector-sharp at any zoom.

- **Zoom:** pinch, or hold Ctrl/⌘ and scroll. You can also click the diagram and then scroll. Until you click it, scrolling moves the chat as usual.
- **Pan and reset:** drag to pan. Double-click zooms in, and double-clicking again fits the diagram back. The `+`, `−`, and `0` keys work too.
- **Toolbar:** the top-right toolbar has zoom controls, full screen, and downloads as **PNG** (2×), **SVG** (fonts embedded), or the original **HTML** file.

Set the `VIEWER` valve to false to show the plain file instead. The saved file is never modified either way.

### Install

1. Clone the skill next to `docker-compose.yaml`. The compose file mounts it read-only into Open WebUI:

   ```bash
   git clone https://github.com/cathrynlavery/diagram-design
   docker compose up -d openwebui
   ```

   This step is optional. Without a clone, the tool fetches the skill files read-only from GitHub at the ref set in its `GITHUB_REF` valve, and it skips the self-check, which only runs from a local clone. Run `git -C diagram-design pull` to update.

2. In Open WebUI, go to **Workspace → Tools → +**, paste the contents of `Tools/diagram_design.py`, and save.
3. Enable the tool on a model (**Workspace → Models → edit → Tools**) or per chat. Use **Native** function calling for best results.
4. Ask for a diagram in plain words, for example: *"Draw an architecture diagram of this stack: Open WebUI, llama.cpp, Postgres/pgvector, SearXNG, and Jupyter."*

Saved diagrams go to `/app/backend/data/diagrams` inside the container (the `openwebui-data` volume). To copy them out:

```bash
docker cp openwebui:/app/backend/data/diagrams ./diagrams
```

The model writes all of the SVG itself, so output quality depends on the model. The skill's references are long: the guide is about 6k tokens, and a type reference plus its example is another 2–8k, so give the model a context window of at least 32k. If context is tight, lower the `MAX_EXAMPLE_CHARS` valve.

## Avatar

`avatar/` is an optional browser app with a talking 3D head. The head is made from one photo with [LAM (Large Avatar Model)](https://github.com/aigc3d/LAM) from Alibaba's Tongyi Lab. The local LLM writes the reply, Kokoro speaks it, and [LAM_Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) turns the audio into lip-sync in real time.

The `avatar` service in `docker-compose.yaml` is an nginx container on port **8702**. It serves `avatar/html/` and proxies llama.cpp, Kokoro, the stt service, the Lens bridge and the Audio2Expression server under one origin, so the page needs no CORS setup. LAM and Audio2Expression themselves run in conda envs on the host, because they compile CUDA extensions against pinned PyTorch and CUDA versions.

Short version, once the conda setup is done:

```bash
bash avatar/scripts/setup-web.sh ~/LAM/output/open_avatar_chat/<name>.zip
docker compose up -d avatar
# then open http://localhost:8702/avatar.html
```

With the optional avatar maker (`avatar/scripts/avatar_maker.py`) running, you can also make new avatars from the page: **Setup → New avatar from a photo** uploads a photo, and LAM turns it into a new head in about a minute. With the `music` service running, you can also ask the avatar to play songs (YouTube), live radio stations or podcasts. Neither the service nor the page needs an account or an API key.

### What it looks like

These are from a phone. The page works the same on a phone as on a desktop browser.

<table>
  <tr>
    <td width="33%"><img src="docs/screenshots/avatar-conversation.jpg" alt="Avatar with the conversation panel showing home commands"></td>
    <td width="33%"><img src="docs/screenshots/avatar-home-lights.jpg" alt="Lens Home tab with a brightness slider for the living room lights"></td>
    <td width="33%"><img src="docs/screenshots/avatar-home-media.jpg" alt="Lens Home tab with media cards for a streaming player and a TV"></td>
  </tr>
  <tr>
    <td valign="top"><b>Talk or type.</b> The head is lit and lip-synced as it answers. The conversation panel under it keeps each request and reply, and a chip for whatever a reply touched. When nothing matches, the avatar says so, and you can just correct yourself (<i>"Sorry, I meant the living room media player"</i>).</td>
    <td valign="top"><b>Home control.</b> <i>"Set the living room to 30 percent"</i> changes the lights through Home Assistant and opens a Home tab in the Lens with a tall slider for them. Drag it to adjust, or tap to switch them on or off. The face shrinks into the space above the panel.</td>
    <td valign="top"><b>Media controls.</b> Players and TVs get a card with only the controls they support: power, play and pause, previous and next, and volume. The card shows what's playing, here a Hulu episode on the living room player.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="25%"><img src="docs/screenshots/avatar-music-lens.jpg" alt="Now playing tab in the Lens with artwork, visualizer and controls"></td>
    <td width="25%"><img src="docs/screenshots/avatar-music-fullscreen.jpg" alt="Full-screen music player with large artwork and a link to YouTube"></td>
    <td width="25%"><img src="docs/screenshots/avatar-music-queue.jpg" alt="Music queue showing now playing and 25 songs up next"></td>
    <td width="25%"><img src="docs/screenshots/avatar-lens-notebook.jpg" alt="Lens tab with a weather briefing from a notebook tool, and the mini player above the face"></td>
  </tr>
  <tr>
    <td valign="top"><b>Music.</b> Ask for a song and it plays from YouTube, on a <b>♪</b> tab in the Lens with the artwork, a live visualizer, a seek bar and controls. Tabs from earlier requests stay along the top.</td>
    <td valign="top"><b>Full screen.</b> Tap the artwork for a large player, with the source and a link to the original video.</td>
    <td valign="top"><b>Queue.</b> Related songs line up after the one you asked for. Tap any of them to jump to it.</td>
    <td valign="top"><b>Notebook tools in the Lens.</b> Results from Lens bridge tools open as their own tab, here a five-page weather briefing with a radar map, paged with the arrows or <b>Jump to…</b>. The music keeps playing in a mini player above the face.</td>
  </tr>
</table>

#### A face made from a photo

The same page with a head that the avatar maker built from a single photo (**Setup → New avatar from a photo**). These were captured with Playwright's WebKit (Safari's engine) at phone size, by typing the requests.

<table>
  <tr>
    <td width="33%"><img src="docs/screenshots/avatar-face-talking.jpg" alt="Photo-made avatar speaking, with its answer in the conversation panel"></td>
    <td width="33%"><img src="docs/screenshots/avatar-face-home-control.jpg" alt="Photo-made avatar after setting the basement game area lights to 40 percent, with the light slider in the Lens"></td>
    <td width="33%"><img src="docs/screenshots/avatar-face-song.jpg" alt="Photo-made avatar playing Fly Me to the Moon, with synced lyrics in the Lens"></td>
  </tr>
  <tr>
    <td valign="top"><b>Talking.</b> <i>"Tell me what you can help me with around the house."</i> It answers out loud, lip-synced by Audio2Expression.</td>
    <td valign="top"><b>Home control.</b> <i>"Set the basement game area lights to 40 percent."</i> Home Assistant dims the real lights, and the Lens opens a slider at 40%.</td>
    <td valign="top"><b>A song.</b> <i>"Play Fly Me to the Moon by Frank Sinatra."</i> It plays with lyrics matched to the singing by the stt service.</td>
  </tr>
</table>

**[avatar/README.md](avatar/README.md)** has the full install: the LAM and Audio2Expression conda envs, the version pins and build fixes they need, Blender export, the avatar maker, hosting, troubleshooting, and credits.

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

Sparkboard uses host `/proc` and the Docker socket for monitoring. Docker socket access is security-sensitive even when mounted read-only (`:ro` doesn't limit the Docker API), so do not expose Sparkboard directly to an untrusted network. The container controls and the model director use the same socket.

### WSL storage monitoring

The default Compose configuration includes:

```text
/mnt/c → C:
/mnt/d → D:
```

These mounts are useful under WSL but should be removed or changed for a native Linux installation.

The `DISK_PATHS` environment variable controls the labels shown by Sparkboard.

### What it reports

<a href="docs/screenshots/sparkboard-overview.png"><img src="docs/screenshots/sparkboard-overview.png" alt="Sparkboard host and GPU overview" width="280"></a>

For each container, Sparkboard tracks CPU %, memory (absolute and % of host), and published ports, for example:

<a href="docs/screenshots/sparkboard-containers.png"><img src="docs/screenshots/sparkboard-containers.png" alt="Sparkboard storage and container list" width="280"></a>

| Container | Published port |
|---|---|
| openwebui | 3000→8080 |
| llama | 8000→8000 |
| comfyui | 8188→8188 |
| kokoro | 8880→8880 |
| jupyter | 8888→8888 |
| searxng | 8080→8080 |
| sparkboard | 9102→9102 |
| avatar | 8702→80 |
| music | 127.0.0.1:8704→8000 (for an nginx outside Compose) |
| tika, playwright, postgres, open-terminal, gitlab-mcp | not published |

On an NVIDIA host it also shows GPU utilization, VRAM allocated, temperature, power draw, and SM/memory clocks over time, plus a per-core CPU load breakdown and network/disk throughput history.

<a href="docs/screenshots/sparkboard-gpu-metrics.png"><img src="docs/screenshots/sparkboard-gpu-metrics.png" alt="Sparkboard GPU metrics" width="280"></a>
<a href="docs/screenshots/sparkboard-system-load.png"><img src="docs/screenshots/sparkboard-system-load.png" alt="Sparkboard CPU, network, and disk load" width="280"></a>

### Container controls

Every container in Sparkboard's list has **Restart** and **Stop** buttons, and a stopped container has **Start**. They are off until you set `SPARKBOARD_CONTROL_KEY` in `.env`. The browser asks for the key once and remembers it, and every action is a POST with an `X-Control-Key` header. Each confirm box says what you're about to lose: stopping Qwen, for example, lists the clients that lose their model.

- **Sparkboard itself** can be restarted (back in about 20 seconds) or stopped. It answers the request first and acts a second later. After a stop, nothing on the page can start it again: run `docker start sparkboard`.
- **Model containers** go through the model director (below). Stop *parks* a model until you press Start, so the director doesn't restart it.
- The nightly update leaves stopped containers stopped (see [Nightly automatic updates](#nightly-automatic-updates)).

## Model director

The model director is an optional part of Sparkboard (`sparkboard/app/director.py`) that runs two local models on one 12 GB GPU. Qwen3.6 35B-A3B is the main model. Inkling 2 Mini is a second, optional one. The director decides which of them is loaded, with how much context and how much of the GPU, based on what you're asking for. Turn it on with `SPARKBOARD_DIRECTOR=1`.

<a href="docs/screenshots/sparkboard-models.png"><img src="docs/screenshots/sparkboard-models.png" alt="Sparkboard Models panel: mode buttons, current layout, each model's state and context, and recent switches" width="280"></a>

### Theory of operation

**One front door.** Clients that want the director's help send requests to Sparkboard instead of to llama.cpp: `http://sparkboard:9102/llm/qwen/v1/...` and `/llm/inkling/v1/...`. A public nginx proxies `/llama/` and `/inkling/` there. For each request the director:

1. Checks the llama.cpp API key itself, so a stranger's request can't trigger a model switch.
2. Estimates the context the request needs: about 3.2 characters per token, with image data left out, plus up to 4k tokens for the answer.
3. Picks a layout that has the model loaded with enough context. If that isn't the current layout, it switches first and **holds the request** until the new layout is healthy. The client sees a slow first token, not an error.
4. Streams the reply straight through and records the request time, so it knows when each model is busy or idle.

**Layouts.** A layout is a set of containers that fits the GPU together. Each one is a compose service with its own context and GPU split.

| Layout | Containers | What you get | VRAM |
|---|---|---|---|
| Both loaded *(default)* | `llama` + `llama-inkling` | Qwen 131k at 28 tok/s, Inkling 32k at ~11.5 tok/s | 10.8 GB |
| Qwen boost | `llama-boost` | Qwen 131k at 41 tok/s | 10.9 GB |
| Qwen long | `llama-long` | Qwen 262k at ~35 tok/s | 11.1 GB |
| Inkling boost | `llama-inkling-fast` | Inkling 32k at ~25 tok/s | 10.4 GB |
| Inkling 64k | `llama-inkling-64k` | Inkling 64k at 20 tok/s | 10.9 GB |
| Inkling 128k | `llama-inkling-128k` | Inkling 128k at 13.5 tok/s | 10.6 GB |

*Measured on an RTX 4070 (12 GB) with 54 GB of RAM under WSL2.*

Every Qwen variant takes the network alias `llama` and host port 8000, and every Inkling variant takes `llama-inkling` and 8001. Clients that talk to llama.cpp directly (Open WebUI, the avatar, Home Assistant) therefore reach whichever variant is up, without any change. They just see a short gap while a switch restarts the model.

**When it switches.**

- **Session boost.** When one model gets 2 requests within 3 minutes, and the other model has been quiet for 5 minutes, it gets its boost layout: the GPU to itself, with as many experts on the GPU as fit. The first request of a session is always answered at once in whatever layout is up. The switch happens *between* requests, when a restart costs the least, and a follow-up that arrives mid-switch waits at the front door.
- **Bigger context.** A prompt that doesn't fit the current context moves the model to the smallest layout that fits it. Prompts are never shrunk to fit.
- **Back to both loaded.** After a quiet spell the director returns to both loaded: 10 minutes for the Qwen layouts, 5 for the Inkling ones.
- **Nothing cut off.** Before stopping a model, the director waits for its in-flight requests to finish, including generations from clients that bypass it. It sees those through llama.cpp's `/slots` endpoint.
- **Yields to ComfyUI.** While ComfyUI has a queue, the director doesn't enter a layout that takes the whole GPU, and it leaves one it's already in.
- **Nightly window.** Between 01:45 and 02:45 local time it holds both loaded, so the nightly `docker compose up -d` never fights a variant for port 8000.
- **Fails safe.** If a layout doesn't come up healthy within 5 minutes, the director goes back to both loaded and stops switching for 30 minutes.
- **Self-repair.** Every 10 seconds it compares the running containers with the current layout and fixes any difference. For example, it stops a variant someone started by hand.

**Modes.** Buttons on the Models panel:

| Mode | Effect |
|---|---|
| Auto | All of the above |
| Both | Keep both loaded, with no boosts and no bigger contexts |
| Qwen boost | Keep Qwen boosted. Inkling's requests still bring both back while they need it |
| Inkling boost | Keep Inkling boosted (Qwen stopped) |

**Parking.** Stopping a model from the containers list parks it. It stays off until you press Start, across restarts of Sparkboard. The remaining model rests in its boost layout instead of both loaded, and requests for the parked model get a "stopped" error.

### What the measurements taught

- **Measure each model alone.** Under WSL2, a GPU allocation that doesn't fit spills into Windows' shared GPU memory instead of failing. `nvidia-smi` then tops out around 11.7–11.9 GB on a 12 GB card, and speed becomes unpredictable: a spilled layout once ran at 2.5 tok/s. So every figure above was taken with one model loaded, and every layout leaves at least 1.2 GB free.
- **An idle model costs nothing.** Qwen runs at the same speed whether or not Inkling is loaded beside it. A model is only stopped when its VRAM buys the other one speed or context.
- **Offload the looped layers.** Inkling 2 Mini has 24 physical layers, but layers 6–17 run three times per token (`inkling2.num_loops` in the GGUF metadata). Putting *those* layers' experts on the GPU with `-ot 'blk\.(6|7|…)\.ffn_.*_exps\.=CUDA0,_exps\.=CPU'` took it from 12 to 26 tok/s. `--n-cpu-moe` offloads from the last layer backwards, and barely helps.
- **Smaller micro-batches save VRAM.** `-ub 256` instead of `512` cuts Inkling's compute buffer by about 1.6 GB. That buys 32k of context beside Qwen, or more experts on the GPU, at a small cost in prompt-processing speed.

### Set it up

1. Download Unsloth's Inkling 2 Mini GGUF (`Inkling-2-Mini-UD-Q4_K_XL.gguf` and its `mmproj-F16.gguf`) into `models/inkling/`.
2. Unpack the CUDA 13 build of Unsloth's llama.cpp fork (`b12588-mix-c0c3de8`) into `opt/llama-unsloth-b12588/`. Mainline llama.cpp doesn't run this model yet.
3. Build and start Inkling, then create the variants:
   ```bash
   docker compose --profile inkling up -d --build llama-inkling
   docker compose --profile variants create llama-boost llama-long \
     llama-inkling-fast llama-inkling-64k llama-inkling-128k
   ```
4. Add to `.env`, then recreate Sparkboard with `docker compose up -d sparkboard`:
   ```bash
   SPARKBOARD_DIRECTOR=1
   SPARKBOARD_CONTROL_KEY=<openssl rand -hex 16>
   ```
5. Point clients at the front door. For opencode, `ops/opencode/opencode.json` defines both models through a public nginx: replace `your-host.example`, copy the file to `~/.config/opencode/opencode.json` on each PC, and set `HOME_LLAMA_API_KEY` there to your `LLAMA_API_KEY`. The config has no `small_model`, so opencode's side requests (such as chat titles) go to the model you're using instead of knocking the other model out of its boost. The routes look like this, and the full Windows nginx config is in [`nginx/windows/`](nginx/windows/):
   ```nginx
   location /llama/   { proxy_pass http://127.0.0.1:9102/llm/qwen/;    proxy_buffering off; proxy_read_timeout 900s; }
   location /inkling/ { proxy_pass http://127.0.0.1:9102/llm/inkling/; proxy_buffering off; proxy_read_timeout 900s; }
   ```
   To bypass the director, point the routes back at `127.0.0.1:8000/` and `127.0.0.1:8001/`.

On a different GPU, re-measure each variant alone and adjust `-c`, `--n-cpu-moe`, `-ot` and `-ub` until each layout leaves 1 GB or more free. The layout table at the top of `director.py` holds the context sizes the director plans with.

### Settings

All are optional environment variables on the `sparkboard` service.

| Variable | Default | Meaning |
|---|---|---|
| `DIRECTOR` | `0` | `1` turns the director on (compose maps `SPARKBOARD_DIRECTOR` to it) |
| `DIRECTOR_BOOST_AFTER` / `DIRECTOR_BOOST_WINDOW` | `2` / `180` | Requests within this many seconds that make a session |
| `DIRECTOR_GRACE` | `300` | How long the other model must have been quiet, in seconds, before a boost |
| `DIRECTOR_ALLOW_QWEN_STOP` | `1` | `0` forbids the Inkling-only layouts, so Qwen always stays up |
| `DIRECTOR_QUIET` | `01:45-02:45` | Local time window held in both loaded |
| `DIRECTOR_HEALTH_TIMEOUT` | `300` | Seconds a layout gets to come up healthy |
| `DIRECTOR_COMFY_URL` | `http://comfyui:8188` | Where to check ComfyUI's queue |
| `DIRECTOR_STATE` | `/data/director.json` | Mode and parked models, kept across restarts |

## Speech-to-text

The `stt` service is [ScribeCrate](https://github.com/hwdsl2/scribecrate) (`hwdsl2/whisper-server`), a faster-whisper server with the OpenAI `/v1/audio/transcriptions` API. Open WebUI's voice mode and dictation, the avatar page and the music service's lyric matching all share it. It takes WebM and M4A straight from browsers and phones, and returns word timestamps in `verbose_json`.

It runs `small.en` on the CPU, which transcribes a spoken turn in under a second and leaves the GPU to llama.cpp and ComfyUI. To change the model, set `WHISPER_MODEL` (for example `large-v3-turbo`, preferably with the `:cuda` image). The server loads one model and ignores the `model` field in requests.

Open WebUI stores audio settings in its database after the first boot. On an existing install, set them in **Admin Panel → Settings → Audio** as well:

| Setting | Value |
|---|---|
| Speech-to-Text Engine | OpenAI |
| API Base URL | `http://stt:8000/v1` |
| API Key | `none` |
| STT Model | `small.en` |

> Earlier versions of this stack used [Speaches](https://github.com/speaches-ai/speaches), which hasn't published a release since v0.9.0-rc.3 (December 2025). On the same clips ScribeCrate was about twice as fast, with identical text and half the memory.

## Ports

| Port | Service | Purpose |
|---:|---|---|
| 3000 | Open WebUI | Main interface |
| 8000 | llama.cpp | OpenAI-compatible model API *(localhost)* |
| 8001 | llama.cpp (Inkling) | Second model's API *(localhost, optional)* |
| 8080 | SearXNG | Search interface *(localhost)* |
| 8090 | stt | Speech-to-text API *(localhost)* |
| 8188 | ComfyUI | Image generation UI |
| 8880 | Kokoro | TTS API *(localhost)* |
| 8888 | Jupyter | Notebook/code environment *(localhost)* |
| 8702 | Avatar | Avatar web app *(optional)* |
| 9102 | Sparkboard | Monitoring dashboard *(localhost)* |

Ports marked *localhost* are bound to `127.0.0.1`. Containers reach each other over the Compose network, and a reverse proxy on the same machine can still use them, but nothing else on the LAN can. Remove the `127.0.0.1:` prefix to open one up.

Playwright, Tika, PostgreSQL, Open Terminal, and GitLab MCP do not publish host ports.

The LAM Audio2Expression server runs on the host at port 8701, and the optional avatar maker at port 8703, both outside Compose. The `avatar` container reaches it through `host.docker.internal`.

## Security

This is a **single-user / trusted-user stack**, not a hardened multi-tenant platform.

In particular, Open Terminal can provide extremely powerful host/container access. Do not expose it directly to the Internet.

The GitLab MCP connection lets the model write to GitLab as the token's user. Scope that user to specific projects, keep `GITLAB_PERMISSION_MODE` at `modify` or `readonly`, and protect `main` so changes go through merge requests.

Before exposing Open WebUI externally:

- use HTTPS
- use strong authentication
- keep secrets out of Git
- keep administrative services bound to localhost (llama.cpp, Jupyter, SearXNG, Kokoro, stt and Sparkboard already are)
- use a long random `LLAMA_API_KEY` and other secrets (`openssl rand -hex 24`), especially if a proxy publishes the llama.cpp API
- understand what host directories are mounted into containers

## Updating

Pull new images by hand:

```bash
docker compose pull
docker compose up -d
```

### Nightly automatic updates

`ops/` has an unattended nightly update. Set it up once:

```bash
sudo bash ops/install-maintenance.sh
```

Every day at 02:00 a systemd timer runs `ops/nightly-update.sh`, which:

1. Backs up the Open WebUI Postgres database and Home Assistant's `/config` to `ops/backups/nightly/` (7 days kept)
2. Upgrades the OS packages (`apt full-upgrade`, keeping your config files)
3. Pulls newer images, rebuilds the local ones (`build --pull`), and recreates whatever changed. Only services that are running (or new in the compose file) are brought up. A container you stopped, or a model variant the director has stopped, stays stopped, and the log lists those containers as "Left stopped"
4. Keeps Playwright's version in step with the one inside Open WebUI (`PLAYWRIGHT_VERSION` in `.env`)
5. Prunes superseded images, and reboots if a package asks for it, but only between 01:00 and 05:59. If the PC was off at 02:00, it doesn't catch up later in the day.

Logs go to `ops/logs/`, and `ops/logs/last-run.txt` lists any failures. Run it by hand with `sudo /usr/local/sbin/stack-nightly`, or with `sudo NO_REBOOT=1 /usr/local/sbin/stack-nightly` to skip the reboot. The installer also caps container logs at 3 × 50 MB, turns on Docker `live-restore` (a Docker upgrade no longer restarts every container) and sets `vm.swappiness=10`.

Services with floating tags (`:latest`, `:main`, `:stable`) update every night. llama.cpp is the exception if you pin a build: pin it while a new release is untested with your model.

**On WSL2**, a reboot stops the distro, and nothing starts it again by default. Run `ops/windows-setup.ps1` once in an administrator PowerShell, then let it ask for your Windows password:

```powershell
powershell -ExecutionPolicy Bypass -File ops\windows-setup.ps1 -Distro Ubuntu-24.04
```

It adds a `wsl-keepalive` task that starts the distro at boot and every 5 minutes if it isn't running. It also adds a `wsl-nightly-update` task at 03:30 (`wsl --update`, then a restart if WSL changed) and sets Windows Update's active hours to 08:00–23:00. Reboots from the Linux side stay off until it has run. Add `-NginxDir C:\nginx` to reload a Windows nginx at the same time.

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
├── sparkboard/              dashboard, container controls, model director (app/director.py)
├── llama-inkling/           CUDA runtime image for the optional Inkling model
├── opt/                     Unsloth's llama.cpp build for Inkling (not committed)
├── nginx/windows/           the Windows nginx in front of the stack (TLS, CloudFront origin, LAN rules)
├── ops/                     nightly OS + image updates, backups, WSL watchdog
│   └── opencode/            opencode client config for both models
└── avatar/
    ├── README.md            LAM / Audio2Expression install and hosting guide
    ├── nginx-avatar.conf    nginx config for the avatar container
    ├── html/                avatar.html, lib/ (renderer), avatars/ (bundles)
    ├── music/               music service: YouTube, radio and podcast search and streaming
    └── scripts/             setup-web.sh, export_avatar.py, a2e_server.py, avatar_maker.py, systemd units
```

Model weights, generated data, secrets, databases, and other machine-specific state should not be committed to Git.

## License

Choose a license appropriate for the code you publish. The repository itself is primarily configuration and integration glue around the respective upstream projects; each upstream component remains subject to its own license and terms.
