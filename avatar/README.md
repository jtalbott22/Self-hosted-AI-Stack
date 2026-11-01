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
   ├── lens/tools, lens/call ──────► Lens bridge      (host:8095)   notebook tools for the LLM
   └── make/jobs ──────────────────► Avatar maker     (host:8703)   photo → new avatar (optional)
   │
   └── lib/lam-renderer.js + avatars/<name>.zip  → renders and animates the head (WebGL)

 Once per face, from Setup in the page or from the command line:
   photo.jpg ──► LAM (conda env `lam`) + Blender ──► avatars/<name>.zip
```

There are three parts, plus an optional fourth:

| Part | Where it runs | What it does |
|---|---|---|
| **LAM** | Host, conda env `lam` | Turns one photo into an animatable Gaussian head and exports a `.zip` bundle. Only needed when you make a new avatar. |
| **LAM_Audio2Expression (A2E)** | Host, conda env `lam_a2e`, port 8701 | Turns speech audio into the facial expression frames that drive the head. Runs all the time. |
| **Web app** | `avatar` container, port 8702 | Serves `avatar.html`, the renderer and the bundles, and proxies llama.cpp, Kokoro, Speaches, the Lens bridge, A2E and the avatar maker. |
| **Avatar maker** *(optional)* | Host, conda env `lam`, port 8703 | Lets you make a new avatar from Setup in the page: upload a photo, and LAM turns it into a bundle in about a minute. See [step 5](#5-make-avatars-from-the-page-optional). |

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
| `tts/dev/tune`, `tts/v1/audio/voices` | Kokoro, `kokoro:8880` | Your own voice, and the list of voices in Setup |
| `a2e/a2e` | A2E on the host, `:8701` | Lip-sync model (lip-sync mode `a2e`, LAM avatars only) |
| `stt/v1/audio/transcriptions` | Speaches on the host, `:8090` | Voice input |
| `lens/tools`, `lens/call` | Lens bridge on the host, `:8095` | Notebook tools |
| `make/jobs` | Avatar maker on the host, `:8703` | New avatar from a photo (step 5) |
| `music/search`, `music/more/...`, `music/stream/...`, `music/prefetch/...`, `music/levels/...`, `music/lyrics/...`, `music/log` | Music service, `music:8000` | Songs, radio and podcasts, the songs that follow, the visualizer, lyrics, and playback reports (step 6) |
| `music/clip` | Music service, `music:8000` | The sound from a video for **Your own voice** |
| `lib/lam-renderer.js` | Served file | LAM renderer (falls back to jsDelivr if missing) |
| `avatars/<name>.zip` | Served file | Avatar bundles |
| `avatar-config.json` | Served file, optional | Site-wide settings (below) |

Opened as a local file, the page calls the ports directly instead (`http://localhost:8000/v1`, `:8880`, `:8701`, `:8090`, `:8095`, `:8703`, `:8704`). Every address can also be changed in Setup. If A2E is down, the page falls back to word-timing lip-sync, so speech still works.

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

Because the page calls its services relatively, Windows nginx has to proxy them under the page's folder: `/avatar/llama/` → `127.0.0.1:8000/`, `/avatar/tts/` → `:8880/`, `/avatar/a2e/` → `:8701/`, `/avatar/stt/` → `:8090/`, `/avatar/lens/` → `:8095/`, and for the avatar maker `/avatar/make/` → `:8703/` with `client_max_body_size 16m`. Use the location blocks in `avatar/nginx-avatar.conf` as a template. Otherwise, set absolute addresses such as `http://localhost:8880/v1` in Setup; the services must then accept cross-origin requests.

### Camera behind the face

**Setup → Background** puts a camera behind the face: the rear camera on a phone makes the screen a window with the avatar standing in front of the room, or the front camera puts you behind it.
- **Background blur** fakes a shallow depth of field, so the face reads as in front of the room. **Background darkness** helps it stand out against a bright scene. While you move either slider, Setup clears so you can see the change.
- The video stays in the browser: it isn't recorded or sent anywhere. A camera needs HTTPS (or `localhost`).
- Phones run one camera at a time. A **Camera** call takes over: with the front camera behind the face, the call's picture is shared; with the rear one, the background comes back when the call ends.
- The face is drawn with a transparent background, so the page's backdrop (or the camera) shows around it. Opening the Lens no longer re-renders the face at a new size: the face is always drawn for the whole window and is scaled and slid into the space above the sheet, in step with it.

### Home-screen icon

When you add the page to a phone's home screen (on an iPhone: Share → **Add to Home Screen**), its icon is a picture of the current face. A few seconds after a LAM face loads, the page copies one frame, crops a square around the head and sets it as the page's `apple-touch-icon` (and the browser tab's icon). It's remembered per avatar, so the next visit has it straight away.
- The phone copies the icon when you add it. Changing the avatar later doesn't change an icon that's already on the home screen: remove it and add it again.
- Only LAM faces make a picture. With any other face the phone uses its own default.

### Your own voice

**Setup → Your own voice** turns a recording of you into the avatar's voice. Record 20 seconds in the page (it shows lines to read), or choose any audio or video file the browser can play, such as a voice memo or a phone video.
- A recording made in the page is taken straight from the mic, with the same settings as listening. On an iPhone the page switches the audio session to record first: it's kept in playback mode for music, and in that mode the mic only hears silence.
- Videos, and any audio file the browser can't decode, are uploaded to the music service, whose ffmpeg sends back the sound (`music/clip`, up to 400 MB, the first five minutes). Phones can't decode every video on the page, an iPhone's HEVC `.mov` above all. Audio files the browser can read stay on the phone.
- The page keeps the 28 seconds with the most talking, trims the quiet ends, evens out the level and sends Kokoro a 24 kHz WAV. Kokoro's voice tuner turns that into a voice in under a second.
- **Hear it** has the avatar say a line in the new voice. Nothing is saved. **Use this voice** saves it on the server as `a_<name>_tuned` and makes it this browser's voice. Kokoro never overwrites a voice, so a name that's taken gets a number (`a_me2_tuned`). To go back, pick another voice in the **Voice** box, which lists your voices first.
- It needs Kokoro-FastAPI 0.9.2 or newer with `ENABLE_INNO_TUNER=true` and `ALLOW_LOCAL_VOICE_SAVING=true`, and the `kokoro-voices` volume to keep saved voices across restarts (all in `docker-compose.yaml`). Docker fills the volume with Kokoro's own voices the first time it starts. To delete a voice: `docker exec kokoro rm /app/api/src/voices/v1_0/a_<name>_tuned.pt`.
- It learns the voice's pitch and tone, not accent or every quirk. A quiet room and your normal voice matter more than length: 10 to 30 seconds of clean talking beats a minute with music or other people in the background. Saved voices speak American English.
- **Who can use it.** Anyone who can reach Kokoro can save voices. Port 8880 and the page's `tts/` path should stay on your LAN or behind a login. If another proxy forwards `/tts/` to Kokoro without a login, block `/tts/dev/tune` there.

## 5. Make avatars from the page (optional)

With the avatar maker running, **Setup → New avatar from a photo** turns a photo into an avatar without a terminal. Pick a photo, give it a name, and click **Make avatar**. The page shows each step, then the face crop LAM used, and loads the new head as your avatar when it's done. It takes about a minute on an RTX 4070.

The maker is [`scripts/avatar_maker.py`](scripts/avatar_maker.py), a small FastAPI server in the `lam` env. It runs `export_avatar.py` (step 2) for each upload, so finish step 2 and copy `export_avatar.py` into `~/LAM` first. It needs no extra packages.

Run it from the repository root:

```bash
conda activate lam
uvicorn avatar_maker:app --app-dir avatar/scripts --host 0.0.0.0 --port 8703 --no-proxy-headers
curl -s http://localhost:8703/health    # {"ok":true,"busy":false,"queued":0}
```

To start it at boot, edit and install [`scripts/lam-maker.service`](scripts/lam-maker.service) the same way as the A2E unit.

**Settings** (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `LAM_DIR` | `~/LAM` | LAM repo with `export_avatar.py` |
| `BLENDER` | `~/software/blender-4.0.2-linux-x64/blender` | Blender 4.x |
| `AVATAR_DIRS` | `avatar/html/avatars` | Where finished ZIPs go. List several folders separated by `:` to publish to more than one web server. |
| `MAKER_ALLOW` | `127.0.0.0/8,::1/128,172.16.0.0/12` | Who may call the maker directly. Everyone else gets `403`, so outside users must come through nginx. |

**How it behaves**

- One avatar is made at a time, and up to 3 more wait in line. A photo can be up to 15 MB. Phone photos are turned upright from their EXIF orientation and scaled to at most 2048 px.
- Names are 1–40 lowercase letters, digits, `-` or `_`. An existing name is refused unless **Replace** is ticked.
- LAM runs under a temporary name, so it never overwrites anything you exported yourself in `~/LAM/output/`. That temporary output is deleted afterwards.
- A finished ZIP is checked for `offset.ply`, `skin.glb`, `animation.glb` and `vertex_order.json` before it's published, then written to each folder with an atomic rename. A visitor never loads a half-written file.
- Errors are explained in the page: no face found, GPU out of memory, a bad file, or a name that's taken. The full LAM log is printed to the maker's output (`journalctl -u lam-maker`).

**API** (what the page calls):

| Request | Response |
|---|---|
| `GET /health` | `{"ok": true, "busy": false, "queued": 0}` |
| `POST /jobs?name=alex&replace=false` with the photo as the raw request body | `202` and the job: `{"id", "name", "state": "queued", "position", ...}` |
| `GET /jobs/<id>` | `{"state": "queued" \| "running" \| "done" \| "error", "step": 0–5, "steps": 5, "message", "error", "avatar": "avatars/alex.zip", "elapsed"}` |
| `GET /jobs/<id>/preview` | PNG of the face crop LAM used, once the job is done |

Errors: `400` for a bad name or an empty body, `409` for a name that's taken or already being made, `413` for a photo over 15 MB, `415` for a file that isn't an image, `429` when the line is full. Finished jobs are forgotten after an hour.

**GPU memory.** Making an avatar needs about 4.5 GB of GPU memory on top of whatever else is running. Next to llama.cpp and A2E on a 12 GB card it peaks at about 11.9 GB. On WSL, Windows can move other processes' GPU memory to system RAM to make room, so llama replies may be slow while an avatar is being made. They're back to normal afterwards.

**Who can use it.** Anyone who can open the page can start GPU jobs and add avatars. Keep port 8702 on your LAN, or put the page behind a login.

## 6. Music, radio and podcasts (optional)

The `music` service in `docker-compose.yaml` (`avatar/music/`) lets you ask the avatar to play something: "play some Fleetwood Mac", "put on KEXP", "play the latest Radiolab". Neither the service nor the page needs an account or an API key.

```bash
docker compose up -d --build music
curl http://localhost:8702/music/health     # through the avatar container
```

| Ask for | Source | What plays |
|---|---|---|
| A song, artist, album, genre or mood | YouTube, through [yt-dlp](https://github.com/yt-dlp/yt-dlp) | The best match of song length (1 to 10 minutes), then up to 25 related songs from its YouTube mix. "lofi" finds a lofi song, not a 12-hour stream; ask for "a 1 hour mix", "the full album" or "a live set" to get the long version |
| A radio station by name, call letters, genre or city | [radio-browser.info](https://www.radio-browser.info/) | The live stream |
| A podcast | Apple's podcast search, then the show's RSS feed | The newest episode, or the one whose title matches what you asked for |

The model gets two tools, `play_music` and `music_control` (pause, resume, next, previous, stop, what's playing). They're built into the page, so they work without the Lens bridge. **Setup → Music** turns them off or points them at another server.

**How it plays.**
- **Pausing and resuming.** Music pauses while the mic is open and while the avatar thinks or speaks, then picks up again when the exchange is over.
- **Controls.** A card under the status pill shows what's playing, with play/pause, next and stop buttons. The same controls work from the lock screen, the car's Now Playing screen and steering-wheel buttons. The artwork pulses while a song is on its way.
- **In the Lens.** Asking for music opens the Lens on a **♪ Now playing** tab with the artwork, visualizer, seek bar, controls, source and queue, alongside any tool results. While that tab is showing, the card steps aside. Close the Lens and the card is back.
- **Visualizer.** A spectrum under the card, folded away with the strip below it (remembered per browser). Tap it for another look: bars, waveform, mirror, peak meter, spectrum line or level meter. The choice is remembered. For songs and podcasts the service works out the spectrum from the file (`music/levels/...`, about 0.5 KB per second of audio) and the page draws it in step with the music, so the audio element is never rerouted through Web Audio, which iOS would stop when the screen locks. Live radio has no file to read: Chrome and Android analyse it as it plays, and on iPhone the bars stay flat.
- **Full screen.** Tap the title (or the artwork in the Lens) for large artwork, a seek bar, previous/next and the source: a link to the YouTube video, the station's website or the episode page, plus who published it.
  - The room moves with the music. Behind everything, the artwork and three glows in its colours swirl slowly and swell with the lows, mids and highs.
  - Motes drift up faster as the music gets louder, and sparks fly out from behind the artwork on the beat. The artwork floats, tilts slightly and pulses with the bass.
  - All of it runs off the same spectrum as the visualizer, so it never touches the audio itself. With **Reduce motion** turned on in the phone's settings, it stays still apart from the glow.
  - **Setup → Music → Now playing effects** picks how much: **Full** (all of the above), **Light** (particles, sparks and the moving artwork, without the colour wash, which is the costly part) or **Off**. **Automatic**, the default, is Full on computers and Light on phones, which are already drawing the avatar. Nothing moves while the music is paused.
- **Lyrics.**
  - For songs, the service looks the lyrics up on [LRCLIB](https://lrclib.net/), a free, open lyrics database with no key, most of it timed line by line (`music/lyrics/...`). The artist and song are worked out from the YouTube title. Of the matches, the one closest in length to the video wins, so the timing lines up.
  - Under the title, three lines scroll by with the one being sung lit up. The same line also shows on the small card.
  - Tap them, or **Lyrics**, for the whole song. Tap a line to play from there. **Sooner** and **Later** nudge the timing by half a second.
  - **Matched to the singing.** LRCLIB's timing is for the record, so a video with a skit before the song (or a pause in the middle) runs ahead of it. With `STT_URL` set on the music service, Speaches listens to each song in 30-second pieces, and the lines are slipped to where the words are actually sung, including a second slip after a pause in the middle.
    - The first lines move about 10 seconds in; a whole song takes about 45 seconds of Speaches' CPU. While it listens, the credit reads "matching to the singing…".
    - A song already on time stops after a few pieces. Results are kept with the lyrics, so a song is only heard once.
    - It needs some lines it can make out: very fast rap, screaming, heavy effects or non-English songs may keep LRCLIB's timing. `docker logs music | grep lyrics:` shows what it heard. Leave `STT_URL` out to turn it off.
  - Songs with untimed lyrics only show them under **Lyrics**. Lookups are cached in the `music-cache` volume.
- **Queue.** The **Queue** button in full screen lists what's playing, what's up next (tap one to jump to it), and the last 50 things played in this browser (tap one to play it again; a song replays by its YouTube id, one request).
- **Moving it.** Drag the card to any corner. The chat bubble moves the same way, by the grip at its top, to a corner or the top or bottom middle. Both remember where you put them.
- **Back from the chat.** Whatever a reply opened stays with it in the chat as a chip: a Lens page or the music it started. Scroll back and tap the chip, or the exchange itself, to bring it back. A Lens page reopens as its tab, and music plays again or, if it's already playing, its tab comes to the front.
- **Same-origin audio.** All audio goes through the service at `music/stream/...`, so the page can play plain-http radio and YouTube audio, which only plays for the address that asked for it. `/stream` only serves results from a recent search; it won't fetch an arbitrary URL.
- **Song cache.** Songs are downloaded whole, repaired with ffmpeg so iPhones can seek in them, and cached in the `music-cache` volume. The cache is capped at `CACHE_GB`, 2 GB by default.
- **How fast it starts.**
  - Most of YouTube's players make a client wait out the length of an ad (5 to 15 seconds) before the audio may be fetched. The service asks YouTube's visionOS player first, which doesn't make it wait, so a song is ready about two seconds after it's chosen. If that player is refused, yt-dlp's usual players are the fallback, with the wait.
  - A search returns as soon as the first song is found, and that song starts downloading at once. The related songs come from a second request (`music/more/...`), so the avatar's answer doesn't wait for them.
  - In practice the music starts within about a second of the avatar finishing its reply.
- **What it downloads.** A search fetches the song list only, no media, plus the first song. Once a song is playing, the next one is looked up (a couple of small requests, no media), so a skip has only the download left, about a second. The next song itself downloads only in the last minute of the current one, so skipping around never pulls down songs you don't hear. At most two songs download at once.

**iPhone and CarPlay.** iOS lets a page start sound only after a tap. Tapping the mic once covers that, and after it the model can start music by voice. Music keeps playing with the screen locked. To stop the end of each reply getting cut off in the car, set **Setup → Listening → Speakers** to **Car** (see Troubleshooting).

**When YouTube changes.** yt-dlp needs regular updates to keep up with YouTube. If songs stop playing but radio and podcasts still work, rebuild the image to get the newest yt-dlp: `docker compose build --no-cache music && docker compose up -d music`.

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
| "The avatar maker isn't reachable" | Check that `curl http://localhost:8703/health` and `curl http://localhost:8702/make/health` answer. The maker must listen on `0.0.0.0` for the container to reach it |
| Maker says `Use the avatar page to make avatars.` (403) | You called port 8703 directly from another machine. Go through the page (port 8702), or add your network to `MAKER_ALLOW` |
| The same 403, or a 502, only through a CDN or reverse proxy (CloudFront, Cloudflare) | Start uvicorn with `--no-proxy-headers`, as the unit does. Without it, uvicorn trusts `X-Forwarded-For` from a proxy on `127.0.0.1`, and behind a CDN that header holds the visitor's public address, which the maker refuses |
| An avatar made outside the maker is missing `skin.glb` | Blender resolves `--python tools/...` against `$PWD`, not the working directory, and exits 0 when the script is missing. Run `export_avatar.py` from a shell whose `$PWD` is the LAM folder (the maker sets this itself) |
| The end of a reply is cut off before the mic opens again, in the car or on Bluetooth | Set **Setup → Listening → Speakers** to **Car** or **Bluetooth**. Wireless audio reaches the speakers a second or more after the page sends it, and opening the mic switches the phone's audio route, which drops whatever is still on its way |
| "The music service isn't reachable" | `curl http://localhost:8702/music/health`. If that fails, `docker compose up -d --build music` and check `docker logs music` |
| Radio and podcasts play but songs don't ("YouTube wouldn't play that") | Update yt-dlp: `docker compose build --no-cache music && docker compose up -d music` |
| Music stutters over wireless CarPlay (or AirPlay) | Most likely the network: streamed, every byte of a song comes from your server as it plays, and over mobile data (through a CDN, from a home connection) an iPhone's player stalls now and then without reporting it. **Setup → Music → Download each song before playing it** (on by default on phones) fetches each song whole, plays it from memory, and fetches the next one during the last minute of the current one; playback reports include `download` (MB, seconds, Mbit/s) and, after each reply, `speech` (how long the phrases took to arrive, in ms). On an iPhone, Safari itself can get into a state where its audio over wireless CarPlay stutters, even for other pages, until it's restarted: swipe the app (or Safari) away and open it again. The page notices when a song plays well behind real time with plenty buffered and says so (`stuck-audio` in the reports). While music plays, the page suspends its speech audio (a running Web Audio graph keeps the phone's audio in small low-latency buffers), and a LAM face is drawn at most at 2x pixel density. If it still stutters, compare with **Setup → Troubleshooting → CarPlay audio test** (when the server has `carplay-test.html`): if a bare audio player stutters too, it's Safari and the phone's link to the car. **Setup → Troubleshooting → Start the guided CarPlay test** (or open the page with `?cartest`) does all of this for you: after one tap on Ready it plays the same song through nine setups, about 75 s each (the plain file, then the downloaded song with everything off, then each part switched back on in turn, then everything on but streamed), asks Smooth or Stutters for each, measures each one, reports `test-step` lines to the music server's log, and puts your settings back. By hand: if the bare player is fine, use **Setup → Troubleshooting → Performance tests** to switch off the face drawing, the visualizer and effects, the lyrics following, tilt parallax, the silent keep-alive loop or the audio mode the page sets ("playback", "record" while listening; off leaves it to iOS) one at a time. The switches stay off on that device until switched back on. Playback reports in `docker logs music \| grep page:` include `"off"` (the Performance tests switched off), `"ac"` (`suspended` while music plays) the pause and play commands that come from the lock screen or the car (`remote-pause`, `remote-play`, `remote-stop`), and every 30 s of playback a `perf` report: `stalls` (times the page's main thread froze for over 150 ms), `worst` (the longest, in ms), `dry` (times the player ran out of audio) and `rate` (seconds of song played per second: under 1 means the phone isn't taking the audio as fast as it should, which is heard as stutter) |
| Songs show up but don't play, or skip on their own | `docker logs music \| grep page:` shows what the phone's audio element did: plays it refused (`play-refused`), stalls, errors with their codes, and skips after errors. If it only happens with the effects on, set **Now playing effects** to **Light** or **Off** |
| Music won't start on an iPhone until you tap ▶ | iOS needs one tap on the page first. Tap the mic or ▶ once and voice requests work from then on |
| Speech starts a second or two after the reply text appears | The page sends the first few words to Kokoro as soon as there's a comma (or a joining word in a long opening), and doesn't wait on the phoneme lookup. The rest is the hardware: with llama's experts on the CPU (`--n-cpu-moe`), the model writes about 15 tokens a second, and Kokoro, also on the CPU, runs about half as fast while the model is writing. Kokoro on the GPU (`kokoro-fastapi-gpu`, under 1 GB of VRAM) or more of llama's expert layers on the GPU both help. A request that uses a tool ("play...") waits for two model replies before it speaks |
| "Voice learning is turned off in Kokoro" or "Kokoro isn't allowed to keep voices" | Add `ENABLE_INNO_TUNER=true` and `ALLOW_LOCAL_VOICE_SAVING=true` to the kokoro service, then `docker compose up -d kokoro` |
| A saved voice is gone after Kokoro was recreated | The `kokoro-voices` volume is missing from the kokoro service, so voices were saved inside the container |
| "Couldn't read the sound in that file", or a video fails | Videos go through the music service: `curl http://localhost:8702/music/health`, and `docker logs music \| grep clip:` shows what ffmpeg said. A 404 means the music image predates `/clip`: `docker compose up -d --build music` |
| A recording in the page says it was silent, or the level bar barely moves | Make sure no other app (a call, a voice memo) has the mic. On an iPhone, reload the page and tap **Record** again; the first tap may only get the permission prompt |
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
