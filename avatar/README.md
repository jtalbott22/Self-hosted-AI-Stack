# Avatar

`avatar.html` is a talking 3D head that runs in the browser. You type or speak, the local LLM answers, Kokoro speaks the answer, and a Gaussian-splat avatar lip-syncs to the audio. The avatar is made from a single photo.

It is built on **LAM (Large Avatar Model)** and its companion projects from Alibaba's Tongyi Lab. See [Credits](#credits).

## How it fits together

```text
 Browser: avatar.html  (http://localhost:8702/avatar.html)
   │
   │  same origin, proxied by the `avatar` nginx container
   ├── llama/v1/chat/completions ──► llama.cpp        (llama:8000)  reply text
   ├── tts/v1/audio/speech ────────► Kokoro           (kokoro:8880) reply audio
   ├── a2e/a2e ────────────────────► Audio2Expression (host:8701)   audio → facial expression frames
   ├── stt/v1/audio/transcriptions ► Speaches         (host:8090)   your voice → text
   └── lens/tools, lens/call ──────► Lens bridge      (host:8095)   notebook tools for the LLM
   │
   └── lib/lam-renderer.js + avatars/<name>.zip  → renders and animates the head (WebGL)

 Offline, once per face:
   photo.jpg ──► LAM (conda env `lam`) + Blender ──► output/open_avatar_chat/<name>.zip
```

There are three parts:

| Part | Where it runs | What it does |
|---|---|---|
| **LAM** | Host, conda env `lam` | Turns one photo into an animatable Gaussian head and exports a `.zip` bundle. Only needed when you make a new avatar. |
| **LAM_Audio2Expression (A2E)** | Host, conda env `lam_a2e`, port 8701 | Turns speech audio into the facial expression frames that drive the head. Runs all the time. |
| **Web app** | `avatar` container, port 8702 | Serves `avatar.html`, the renderer and the bundles, and proxies llama.cpp, Kokoro, Speaches, the Lens bridge and A2E. |

LAM and A2E compile CUDA extensions against specific versions of PyTorch and CUDA, so they run in conda envs on the host, not in Compose.

## Requirements

- The main stack running (`llama` and `kokoro` at least; `speaches` for voice input and `lens-bridge` for notebook tools).
- An NVIDIA GPU with CUDA 12.1 support. The steps below were done on WSL2 Ubuntu with an RTX 40-series card (`TORCH_CUDA_ARCH_LIST=8.9`). For a 30-series card use `8.6`. CUDA 12.1 does not support RTX 50-series (Blackwell) cards.
- About 25 GB of free disk for the two conda envs, the model weights and Blender.

## 1. System packages and conda

```bash
sudo apt update
sudo apt install -y gcc-12 g++-12 build-essential git ffmpeg zip \
  libxi6 libxrender1 libxkbcommon0 libsm6 libgl1 libxxf86vm1 libxfixes3
```

gcc-12 is needed because CUDA 12.1's `nvcc` rejects newer host compilers. The X libraries are for headless Blender.

Install Miniconda, accept the defaults, then open a new shell:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

New conda versions refuse to create envs until you accept the channel terms:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
```

## 2. LAM (avatar generation)

### Create the env

```bash
conda create -n lam python=3.10 -y
conda activate lam
```

Install the CUDA 12.1 toolkit so `nvcc` is inside the env. Use the versioned label. Plain `cuda-toolkit=12.1` does not pin the toolkit correctly.

```bash
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit -y
```

Save the compiler settings in the env, so every later `conda activate lam` has them. If you only `export` these, a new shell loses them and the CUDA extensions build with the wrong compiler.

```bash
conda env config vars set -n lam CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12 TORCH_CUDA_ARCH_LIST=8.9
conda deactivate && conda activate lam
echo $CC   # gcc-12
```

### Install LAM

```bash
cd ~
git clone https://github.com/aigc3d/LAM.git
cd LAM

# Build tools the install script assumes are already there
pip install cython "numpy==1.26.4" setuptools wheel ninja

bash ./scripts/install/install_cu121.sh
```

If the script stops at pytorch3d or at `requirements.txt`, install those two by hand without build isolation, so they build against the torch already in the env:

```bash
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
pip install --no-build-isolation -r requirements.txt "numpy<2"
```

Fix the FaceBoxesV2 NMS extension. `np.int_t` no longer exists in current Cython/NumPy:

```bash
sed -i 's/np\.int_t/np.intp_t/g' external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms.pyx
(cd external/landmark_detection/FaceBoxesV2/utils && python build.py build_ext --inplace)
```

Pin the web and Hugging Face packages. LAM's Gradio app breaks with current releases of these, and each one fails in turn with a different error. These are the versions that work together:

```bash
pip install "transformers==4.44.2" "huggingface_hub<1.0" \
  "fastapi==0.104.1" "starlette<0.28" "pydantic==2.5.3" \
  "websockets<12" "uvicorn==0.23.2" "httpx==0.24.1" "numpy<2"
```

Check the result:

```bash
python -c "import torch, pytorch3d, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
# torch 2.x+cu121, True, 1.26.x
```

### Download the LAM weights

```bash
cd ~/LAM
hf download 3DAIGC/LAM-assets --local-dir ./tmp
tar -xf ./tmp/LAM_assets.tar && rm ./tmp/LAM_assets.tar
tar -xf ./tmp/thirdparty_models.tar && rm -r ./tmp/
hf download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/

# The asset tarballs contain macOS "._*" metadata files that break some loaders
find ~/LAM/assets -name "._*" -delete
```

`hf` is the current Hugging Face CLI. Older guides use `huggingface-cli download`, which does the same thing.

### Set up export (FBX, Blender, sample motion)

The `.zip` bundle that the web renderer loads is written by LAM's export path, which needs the FBX SDK, a sample animated character and Blender 4.0:

```bash
cd ~/LAM
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl
pip install fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl patool "numpy<2"

wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/sample_oac.tar
tar -xf sample_oac.tar -C assets/
ls assets/sample_oac

mkdir -p ~/software && cd ~/software
wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
tar -xf blender-4.0.2-linux-x64.tar.xz
~/software/blender-4.0.2-linux-x64/blender --background --version
```

LAM's Gradio app fails on export when no driving video is chosen in the UI. This patch falls back to the bundled GEM sample motion (or `$LAM_MOTION` if set):

```bash
cd ~/LAM
sed -i 's|^        base_vid = os.path.basename(video_params).split(".")\[0\]|        video_params = video_params or os.environ.get("LAM_MOTION", "./assets/sample_motion/export/GEM/GEM.mp4")\n&|' app_lam.py
grep -n "LAM_MOTION" app_lam.py   # one match
```

Optional: show Python errors in the browser instead of a bare "Error":

```bash
sed -i 's/demo.launch()/demo.launch(debug=True, show_error=True)/' app_lam.py
```

### Make an avatar

Use a well-lit, front-facing photo with the whole head in frame.

**Gradio UI:**

```bash
cd ~/LAM && conda activate lam
python app_lam.py --blender_path ~/software/blender-4.0.2-linux-x64/blender
```

Open the URL it prints (Gradio defaults to `http://localhost:7860`), upload the photo, generate, and export. The bundle is written to `~/LAM/output/open_avatar_chat/`.

**Command line:** [`scripts/export_avatar.py`](scripts/export_avatar.py) runs the same export as the Gradio app's "Export ZIP file for Chatting Avatar" button, without the UI. It isn't part of upstream LAM, so copy it into the LAM repo root and run it from there:

```bash
cp avatar/scripts/export_avatar.py ~/LAM/      # from this repo's root
cd ~/LAM && conda activate lam
python export_avatar.py ~/photos/josh.jpg --blender ~/software/blender-4.0.2-linux-x64/blender --name josh
```

| Argument | Meaning |
|---|---|
| `photo` | The input photo (any format Pillow reads). |
| `--blender` | Path to the Blender 4.x executable. Required. |
| `--name` | Avatar name, used for the ZIP and its top folder. Default: the photo's file name without extension. |
| `--motion` | Sample motion folder under `assets/sample_motion/export/` that is used for the internal render pass. Default: `GEM`. It doesn't change the exported avatar's animation. |

It checks for the photo, Blender, the motion folder and `assets/sample_oac/` (`template_file.fbx`, `animation.glb`) before loading the model, and exits with the missing path if one isn't there. It writes:

- `output/open_avatar_chat/<name>.zip`, one `<name>/` folder holding `offset.ply`, `skin.glb`, `animation.glb` and `vertex_order.json`. This is the file the page loads.
- `output/open_avatar_chat/<name>/`, the same files unzipped. It is replaced on every run with the same name.
- `output/open_avatar_chat/<name>_processed.png`, the cropped, matted face LAM actually used. Check it if the avatar looks wrong.

The script calls LAM's own functions directly, so the `LAM_MOTION` patch to `app_lam.py` above isn't needed for it.

If export fails while compiling `nvdiffrast` (for example, because `CC` wasn't set the first time), clear the cached build and run it again:

```bash
rm -rf ~/.cache/torch_extensions/*/nvdiffrast_plugin
```

## 3. LAM_Audio2Expression (lip-sync server)

A2E gets its own env because its dependency pins conflict with LAM's.

```bash
cd ~
git clone https://github.com/aigc3d/LAM_Audio2Expression.git
cd LAM_Audio2Expression

conda create -n lam_a2e python=3.10 -y
conda activate lam_a2e
bash ./scripts/install/install_cu121.sh
pip install fastapi "uvicorn[standard]"

hf download 3DAIGC/LAM_audio2exp --local-dir ./
tar -xzvf LAM_audio2exp_assets.tar && rm -f LAM_audio2exp_assets.tar
tar -xzvf LAM_audio2exp_streaming.tar && rm -f LAM_audio2exp_streaming.tar
```

Upstream A2E has no HTTP server, so this repo has a small FastAPI wrapper, [`scripts/a2e_server.py`](scripts/a2e_server.py). Copy it into the A2E repo root and run it from there on port 8701:

```bash
cp avatar/scripts/a2e_server.py ~/LAM_Audio2Expression/   # from this repo's root
cd ~/LAM_Audio2Expression && conda activate lam_a2e
uvicorn a2e_server:app --host 0.0.0.0 --port 8701
```

It loads the streaming model (`configs/lam_audio2exp_config_streaming.py`; set `A2E_CONFIG` to use another config) and runs one warm-up pass before it accepts requests, so the first real request isn't slow. It needs a CUDA GPU and runs one request at a time on it.

**API**

| Request | Response |
|---|---|
| `GET /health` | `{"ok": true, "fps": 30, "styles": 12, "gpu": "<GPU name>"}` |
| `POST /a2e?style=0` with the audio file as the raw request body | `{"fps": 30, "names": [52 ARKit blendshape names], "frames": [[52 floats], ...], "duration": <seconds>, "infer_ms": <ms>}` |

- The body can be any format librosa can decode: WAV, MP3, OGG or FLAC. It is resampled to 16 kHz mono. The `Content-Type` header is ignored; the page sends Kokoro's MP3 as `application/octet-stream`.
- `style` (optional, default 0) picks one of the model's speaking styles, `0` to `styles - 1` (0–11 with the released weights).
- There are 30 frames per second of audio, each with 52 weights from 0 to 1, in the order given by `names`. Frame `i` belongs to time `i / fps`. Weights are pulled toward a closed mouth during silence.
- Errors: `400` for an empty body or a `style` out of range, `413` for a body over 20 MB, `415` when the audio can't be decoded.
- CORS is open (`*`), so a page on another origin can call it directly.

Test it with one of the bundled WAV files:

```bash
curl -s http://localhost:8701/health
cd ~/LAM_Audio2Expression
WAV=$(find assets -name "*.wav" | head -1); echo $WAV
curl -s -X POST --data-binary @"$WAV" -H "Content-Type: application/octet-stream" \
  http://localhost:8701/a2e | head -c 300
```

To start it at boot (WSL needs `systemd=true` under `[boot]` in `/etc/wsl.conf`), edit and install the unit in `avatar/scripts/lam-a2e.service`. The steps are in the comments at the top of the file.

## 4. Host the web app

From the repository root:

```bash
# 1. Renderer: downloads gaussian-splat-renderer-for-lam into avatar/html/lib/
#    and copies your avatar bundle(s) into avatar/html/avatars/
bash avatar/scripts/setup-web.sh ~/LAM/output/open_avatar_chat/josh.zip

# 2. Serve avatar/html/ (avatar.html is already in the repo)
docker compose up -d avatar
```

Open **http://localhost:8702/avatar.html**, then open Setup and set the avatar to `avatars/josh.zip`.

The `avatar` container serves `avatar/html/` read-only, so new bundles and edits to `avatar.html` show up on reload without a restart.

### Endpoints the page uses

When the page is served over http or https, its defaults are paths **relative to the page**, and the container proxies each one, stripping the prefix. The page needs no CORS settings and no hard-coded ports, and it works the same from other machines on your LAN or under a sub-path behind a reverse proxy.

| Default in the page | Proxied to | Used for |
|---|---|---|
| `llama/v1` + `/chat/completions` | llama.cpp, `llama:8000` | Replies (streamed) |
| `tts/v1` + `/audio/speech` | Kokoro, `kokoro:8880` | Reply audio |
| `tts/dev/captioned_speech`, `tts/dev/phonemize` | Kokoro, `kokoro:8880` | Word timings and phonemes for the built-in lip-sync |
| `a2e/a2e` | A2E on the host, `:8701` | Lip-sync model (lip-sync mode `a2e`, LAM avatars only) |
| `stt/v1/audio/transcriptions` | Speaches on the host, `:8090` | Voice input |
| `lens/tools`, `lens/call` | Lens bridge on the host, `:8095` | Notebook tools |
| `lib/lam-renderer.js` | Served file | LAM renderer (falls back to jsDelivr if missing) |
| `avatars/<name>.zip` | Served file | Avatar bundles |
| `avatar-config.json` | Served file, optional | Site-wide settings (below) |

Opened as a local file, the page calls the ports directly instead (`http://localhost:8000/v1`, `:8880`, `:8701`, `:8090`, `:8095`). Every address can also be changed in Setup. If A2E is down, the page falls back to word-timing lip-sync, so speech still works.

**API keys.** If llama runs with `LLAMA_API_KEY`, or the Lens bridge with `LENS_BRIDGE_KEY`, requests without the key get `401`. Either enter the keys in Setup (they stay in that browser), or set them on the server with `proxy_set_header Authorization "Bearer ...";` in the `/llama/` or `/lens/` block of `avatar/nginx-avatar.conf`. With the second option, anyone who can reach port 8702 can use them.

**Site settings.** Setup can export your settings as `avatar-config.json`. Put it in `avatar/html/` and every visitor starts from those settings. It is gitignored, because the export can include the keys.

### No avatar yet? Use the A2E sample

A2E ships with a sample head you can use to test the page before generating your own:

```bash
D=$(dirname "$(find ~/LAM_Audio2Expression/assets -name offset.ply | head -1)")
(cd "$(dirname "$D")" && zip -r ~/sample-avatar.zip "$(basename "$D")")
unzip -l ~/sample-avatar.zip   # one folder with the bundle files inside
bash avatar/scripts/setup-web.sh ~/sample-avatar.zip
```

### Alternative: Windows nginx

If you already run nginx on Windows (for example, `C:\nginx`), copy the folder there instead of using the container:

```bash
mkdir -p /mnt/c/nginx/html/avatar
cp -r avatar/html/* /mnt/c/nginx/html/avatar/
```

Because the page calls its services relatively, Windows nginx has to proxy them under the page's folder: `/avatar/llama/` → `127.0.0.1:8000/`, `/avatar/tts/` → `:8880/`, `/avatar/a2e/` → `:8701/`, `/avatar/stt/` → `:8090/`, `/avatar/lens/` → `:8095/`. Use the location blocks in `avatar/nginx-avatar.conf` as a template. Otherwise, set absolute addresses such as `http://localhost:8880/v1` in Setup; the services must then accept cross-origin requests.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `conda create` fails with a Terms of Service error | `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` |
| `nvcc` not found, or the wrong CUDA version, during install | `conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit -y` in the active env |
| `unsupported GNU version` from nvcc | gcc-12 isn't in use. Set the `conda env config vars` from step 2, then deactivate and activate the env again |
| pytorch3d build can't find torch | `pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"` |
| `cpu_nms.pyx`: `np.int_t` error | The `sed` fix to `np.intp_t`, then rebuild FaceBoxesV2 |
| Gradio import or startup errors (`transformers`, `huggingface_hub`, `pydantic`, `starlette`, `websockets`, `httpx`) | Install the pinned set from step 2 |
| Anything mentions NumPy 2 | Add `"numpy<2"` to whatever you just installed. pip upgrades NumPy as a side effect |
| Export: `base_vid` / `NoneType` error | Apply the `LAM_MOTION` patch to `app_lam.py` |
| Blender: `libXi.so.6` (or similar) not found | Install the X libraries from step 1 |
| `nvdiffrast_plugin` build error | `rm -rf ~/.cache/torch_extensions/*/nvdiffrast_plugin` and retry with `CC=gcc-12` set |
| Page loads but the head never moves, or "Lip-sync model unreachable" | Check that `curl http://localhost:8701/health` and `curl http://localhost:8702/a2e/health` both answer, and that `docker logs avatar` shows no 502s on `/a2e/`. On native Linux, A2E must listen on `0.0.0.0`, not `127.0.0.1`, for the container to reach it |
| Replies or tools fail with 401 | llama or the Lens bridge needs its key. See **API keys** under step 4 |
| Renderer fails to load | Check that `avatar/html/lib/lam-renderer.js` exists and has no `import` lines (`grep -c '^import'` prints 0) |

## Credits

The avatar pipeline is built on open research from **Alibaba Group's Tongyi Lab (3DAIGC)**. Many thanks to the authors.

- **[LAM: Large Avatar Model for One-shot Animatable Gaussian Head](https://github.com/aigc3d/LAM)** (SIGGRAPH 2025). It generates the animatable Gaussian head from one image and exports the bundle. Project page: <https://aigc3d.github.io/projects/LAM/>. Weights: [3DAIGC/LAM-20K](https://huggingface.co/3DAIGC/LAM-20K), [3DAIGC/LAM-assets](https://huggingface.co/3DAIGC/LAM-assets).
- **[LAM_Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression)**: audio-driven facial expressions for LAM avatars. Weights: [3DAIGC/LAM_audio2exp](https://huggingface.co/3DAIGC/LAM_audio2exp).
- **[gaussian-splat-renderer-for-lam](https://www.npmjs.com/package/gaussian-splat-renderer-for-lam)**: the WebGL renderer that plays LAM avatars in the browser.
- **[OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat)**: the `open_avatar_chat` bundle format that LAM exports and the renderer loads.
- Also used: [FLAME](https://flame.is.tue.mpg.de/) head model (via the LAM assets), [Blender](https://www.blender.org/), [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [nvdiffrast](https://github.com/NVlabs/nvdiffrast) and [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI).

If you use LAM in research, please cite the paper. The BibTeX is in the [LAM README](https://github.com/aigc3d/LAM#citation).

Each project, model and dataset above has its own license and terms. In particular, the FLAME model and some of the bundled assets are licensed for **non-commercial use only**. Check them before you use generated avatars for anything beyond personal use. Only make avatars of people who have agreed to it.
