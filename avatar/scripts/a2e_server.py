"""
a2e_server.py - HTTP wrapper around LAM Audio2Expression.

Place in the root of the LAM_Audio2Expression repo and run from there:
    uvicorn a2e_server:app --host 0.0.0.0 --port 8701

Endpoints
    GET  /health            -> {"ok": true, ...}
    POST /a2e?style=0       body = raw audio bytes (wav, mp3, ogg, flac)
                            -> {"fps": 30, "names": [52 ARKit names],
                                "frames": [[52 floats], ...], "duration": s,
                                "infer_ms": n}

Every frame lines up with audio time t = index / fps, so the browser can
look up frames[Math.floor(audio.currentTime * fps)] while the clip plays.
"""
import os
import math
import tempfile
import threading
import time

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from engines.defaults import default_config_parser, default_setup
from engines.infer import INFER
from models.utils import ARKitBlendShape

CONFIG = os.environ.get("A2E_CONFIG", "configs/lam_audio2exp_config_streaming.py")
SR = 16000
FPS = 30
MIN_SECONDS = 0.5          # wav2vec needs a little audio to work with
MAX_BYTES = 20 * 1024 * 1024

cfg = default_config_parser(CONFIG, None)
cfg = default_setup(cfg)
infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
infer.model.eval()
NUM_STYLES = cfg.model.backbone.num_identity_classes
_gpu_lock = threading.Lock()


def run_a2e(audio: np.ndarray, style: int) -> np.ndarray:
    """audio: mono float32 at 16 kHz -> (frames, 52) blendshape weights."""
    n_frames = max(1, math.ceil(len(audio) / SR * FPS))
    padded = audio
    if len(audio) < SR * MIN_SECONDS:
        padded = np.concatenate([audio, np.zeros(int(SR * MIN_SECONDS) - len(audio), np.float32)])

    with _gpu_lock, torch.no_grad():
        inp = {
            "id_idx": F.one_hot(torch.tensor(style), NUM_STYLES).cuda()[None, ...],
            "input_audio_array": torch.from_numpy(padded).float().cuda()[None, ...],
        }
        exp = infer.model(inp)["pred_exp"].squeeze(0).cpu().numpy()

    # Per-frame loudness, used to close the mouth during silences.
    hop = SR // FPS
    vol = librosa.feature.rms(y=padded, frame_length=hop, hop_length=hop)[0]
    if len(vol) < len(exp):
        vol = np.pad(vol, (0, len(exp) - len(vol)))
    vol = vol[: len(exp)]

    exp = infer.apply_expression_postprocessing(exp, audio_volume=vol)
    return np.clip(exp[:n_frames], 0.0, 1.0)


app = FastAPI(title="LAM Audio2Expression")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def warmup():
    t = time.time()
    run_a2e(np.zeros(SR, np.float32), 0)
    print(f"[a2e] warm, {1000 * (time.time() - t):.0f} ms", flush=True)


@app.get("/health")
def health():
    return {"ok": True, "fps": FPS, "styles": NUM_STYLES,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@app.post("/a2e")
async def a2e(request: Request, style: int = 0):
    body = await request.body()
    if not body:
        raise HTTPException(400, "Send the audio file as the request body.")
    if len(body) > MAX_BYTES:
        raise HTTPException(413, "Audio is larger than 20 MB.")
    if not 0 <= style < NUM_STYLES:
        raise HTTPException(400, f"style must be 0-{NUM_STYLES - 1}.")

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as f:
        f.write(body)
        f.flush()
        try:
            audio, _ = librosa.load(f.name, sr=SR, mono=True)
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}")

    t = time.time()
    frames = run_a2e(audio.astype(np.float32), style)
    return {
        "fps": FPS,
        "names": ARKitBlendShape,
        "frames": np.round(frames, 4).tolist(),
        "duration": len(audio) / SR,
        "infer_ms": round(1000 * (time.time() - t)),
    }
