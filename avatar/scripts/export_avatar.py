#!/usr/bin/env python
"""
export_avatar.py - turn one photo into a chat-avatar ZIP with LAM, no Gradio.

Mirrors the "Export ZIP file for Chatting Avatar" path in app_lam.py using the
repo's own functions. Run from the LAM repo root in the `lam` conda env:

    python export_avatar.py ~/photos/josh.jpg \
        --blender ~/software/blender-4.0.2-linux-x64/blender --name josh

Output: output/open_avatar_chat/<name>.zip containing one folder with
offset.ply, skin.glb, animation.glb and vertex_order.json.
"""
import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser(description="LAM photo -> chat avatar ZIP")
ap.add_argument("photo", help="front-facing head-and-shoulders photo")
ap.add_argument("--blender", required=True, help="path to the Blender 4.x executable")
ap.add_argument("--name", help="avatar name (default: photo file name)")
ap.add_argument("--motion", default="GEM",
                help="sample motion folder used for the internal render pass (default: GEM)")
args = ap.parse_args()

if not os.path.exists("app_lam.py"):
    sys.exit("Run this from the LAM repo root (the folder with app_lam.py).")

photo = os.path.abspath(os.path.expanduser(args.photo))
blender = os.path.abspath(os.path.expanduser(args.blender))
name = args.name or os.path.splitext(os.path.basename(photo))[0]
motion_dir = f"./assets/sample_motion/export/{args.motion}/flame_param"
template_fbx = "./assets/sample_oac/template_file.fbx"
anim_glb = "./assets/sample_oac/animation.glb"

for path, what in [(photo, "photo"), (blender, "Blender"), (motion_dir, "motion folder"),
                   (template_fbx, "template FBX (extract sample_oac.tar into assets/)"),
                   (anim_glb, "animation.glb (extract sample_oac.tar into assets/)")]:
    if not os.path.exists(path):
        sys.exit(f"Missing {what}: {path}")

# app_lam.parse_configs() reads sys.argv and these env vars, so set both first.
sys.argv = [sys.argv[0], "--blender_path", blender]
os.environ.update({
    "APP_ENABLED": "1",
    "APP_MODEL_NAME": "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
    "APP_INFER": "./configs/inference/lam-20k-8gpu.yaml",
    "APP_TYPE": "infer.lam",
    "NUMBA_THREADING_LAYER": "omp",
})

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app_lam import (  # noqa: E402
    FlameTrackingSingleImage, _build_model, parse_configs,
    prepare_motion_seqs, preprocess_image,
)
from tools.generateARKITGLBWithBlender import generate_glb  # noqa: E402

print(f"[1/5] Loading LAM for '{name}'", flush=True)
cfg, _ = parse_configs()
lam = _build_model(cfg).to("cuda").eval()
tracker = FlameTrackingSingleImage(
    output_dir="output/tracking",
    alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
    vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
    human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
    facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth",
    detect_iris_landmarks=False,
)

work = tempfile.mkdtemp(prefix="lam_export_")
raw = os.path.join(work, "raw.png")
with Image.open(photo) as im:
    im.convert("RGB").save(raw)

print("[2/5] Fitting FLAME to the photo", flush=True)
assert tracker.preprocess(raw) == 0, "FLAME preprocess failed (is a face clearly visible?)"
assert tracker.optimize() == 0, "FLAME optimize failed"
rc, track_dir = tracker.export()
assert rc == 0, "FLAME export failed"

image_path = os.path.join(track_dir, "images/00000_00.png")
mask_path = os.path.join(track_dir, "fg_masks/00000_00.png")
image, _, _, shape_param = preprocess_image(
    image_path, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=1.,
    max_tgt_size=None, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
    render_tgt_size=cfg.source_size, multiply=14, need_mask=True, get_shape_param=True)
Image.fromarray((image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) \
    .save(os.path.join(work, "processed.png"))

print("[3/5] Building the Gaussian head", flush=True)
motion_seq = prepare_motion_seqs(
    motion_dir, None, save_root=work, fps=30, bg_color=1., aspect_standard=1.0,
    enlarge_ratio=[1.0, 1, 0], render_image_res=cfg.render_size, multiply=16,
    need_mask=False, vis_motion=False, shape_param=shape_param, test_sample=False,
    cross_id=False, src_driven=[image_path.split("/")[-3], motion_dir.split("/")[-2]])
motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
with torch.no_grad():
    res = lam.infer_single_view(
        image.unsqueeze(0).to("cuda", torch.float32), None, None,
        render_c2ws=motion_seq["render_c2ws"].to("cuda"),
        render_intrs=motion_seq["render_intrs"].to("cuda"),
        render_bg_colors=motion_seq["render_bg_colors"].to("cuda"),
        flame_params={k: v.to("cuda") for k, v in motion_seq["flame_params"].items()})

print("[4/5] Exporting skin and splats (Blender)", flush=True)
out_root = os.path.abspath("output/open_avatar_chat")
avatar_dir = os.path.join(out_root, name)
shutil.rmtree(avatar_dir, ignore_errors=True)
os.makedirs(avatar_dir)
head_mesh = lam.renderer.flame_model.save_shaped_mesh(shape_param.unsqueeze(0).cuda(), fd=avatar_dir)
res["cano_gs_lst"][0].save_ply(os.path.join(avatar_dir, "offset.ply"), rgb2sh=False, offset2xyz=True)
generate_glb(input_mesh=Path(head_mesh), template_fbx=Path(template_fbx),
             output_glb=Path(avatar_dir) / "skin.glb", blender_exec=Path(blender))
shutil.copy(anim_glb, os.path.join(avatar_dir, "animation.glb"))
os.remove(head_mesh)

print("[5/5] Packing ZIP", flush=True)
zip_path = os.path.join(out_root, f"{name}.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr(f"{name}/", "")                      # the renderer looks for one top folder
    for fn in sorted(os.listdir(avatar_dir)):
        zf.write(os.path.join(avatar_dir, fn), arcname=f"{name}/{fn}")
        print(f"      {name}/{fn}")
shutil.copy(os.path.join(work, "processed.png"), os.path.join(out_root, f"{name}_processed.png"))
print(f"\nDone: {zip_path}\nCrop LAM used: {os.path.join(out_root, name + '_processed.png')}")
