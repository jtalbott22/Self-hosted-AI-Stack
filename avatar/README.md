# Avatar

`avatar.html` is a talking 3D head that runs in the browser. You type or speak, the local LLM answers, Kokoro speaks the answer, and a Gaussian-splat avatar lip-syncs to the audio. The avatar is made from a single photo.

It is built on **LAM (Large Avatar Model)** and its companion projects from Alibaba's Tongyi Lab. See [Credits](#credits).

## How it fits together

```text
 Browser: avatar.html  (http://localhost:8702/avatar.html)
   │
   │  same origin, proxied by the `avatar` nginx container
   ├── /v1/chat/completions ──► llama.cpp       (llama:8000)        reply text
   ├── /v1/audio/speech ──────► Kokoro          (kokoro:8880)       reply audio
   └── /a2e ──────────────────► Audio2Expression (host:8701)        audio → facial expression frames
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
| **Web app** | `avatar` container, port 8702 | Serves `avatar.html`, the renderer and the bundles, and proxies the other services. |

LAM and A2E compile CUDA extensions against specific versions of PyTorch and CUDA, so they run in conda envs on the host, not in Compose.

## Requirements

- The main stack running (`llama` and `kokoro` at least).
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

**Command line:** the `export_avatar.py` helper runs the same steps headless:

```bash
cd ~/LAM && conda activate lam
python export_avatar.py josh.jpg --blender ~/software/blender-4.0.2-linux-x64/blender --name josh
# -> output/open_avatar_chat/josh.zip
```

> `export_avatar.py` is a local helper script and is not in upstream LAM yet. Add it to `avatar/scripts/` in this repo and copy it into `~/LAM` before running.

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

Run the HTTP wrapper on port 8701:

```bash
uvicorn a2e_server:app --host 0.0.0.0 --port 8701
```

> `a2e_server.py` is a small FastAPI wrapper around A2E inference, not part of upstream. It takes raw WAV bytes on `POST /a2e` (`Content-Type: application/octet-stream`) and returns the expression frames as JSON. Add it to `avatar/scripts/` in this repo and copy it into `~/LAM_Audio2Expression`.

Test it with one of the bundled WAV files:

```bash
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

# 2. The page itself
cp /path/to/avatar.html avatar/html/avatar.html

# 3. Serve it
docker compose up -d avatar
```

Open **http://localhost:8702/avatar.html**.

The `avatar` container serves `avatar/html/` read-only, so new bundles and edits to `avatar.html` show up on reload without a restart.

### Endpoints the page should use

The container proxies the backends under its own origin, so the page needs no CORS settings and no hard-coded ports:

| Path in the page | Proxied to |
|---|---|
| `/v1/chat/completions` | llama.cpp, `llama:8000` |
| `/v1/audio/speech` | Kokoro, `kokoro:8880` |
| `/a2e` | A2E on the host, `:8701` |
| `lib/lam-renderer.js` | Served file |
| `avatars/<name>.zip` | Served file |

To see which URLs your copy of `avatar.html` calls:

```bash
grep -nE "audio/speech|/v1|/a2e|localhost:[0-9]+" -i avatar/html/avatar.html
```

Absolute URLs like `http://localhost:8880/v1/audio/speech` still work from a browser on the same machine. Relative paths also work from other machines on your LAN and behind a reverse proxy.

If you set `LLAMA_API_KEY`, the browser has to send it. In that case, either add it in the page or set it in nginx with `proxy_set_header Authorization "Bearer ...";` in the `/v1/` block of `avatar/nginx-avatar.conf`.

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

That gives you `http://localhost/avatar/avatar.html`. Windows nginx doesn't proxy the backends, so the page has to call them by absolute URL (`http://localhost:8880`, `http://localhost:8000`, `http://localhost:8701`). Those services must also accept cross-origin requests.

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
| Page loads but the head never moves | Check that `curl http://localhost:8701/a2e` answers and `docker logs avatar` shows no 502s on `/a2e`. On native Linux, A2E must listen on `0.0.0.0`, not `127.0.0.1`, for the container to reach it |
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
