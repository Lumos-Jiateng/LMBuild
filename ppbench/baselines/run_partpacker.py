#!/usr/bin/env python3
"""PartPacker (NVlabs/PartPacker): one RGB image -> dual-volume latent -> connected-component part meshes.
Standalone; imports nothing from ppbench.

Mirrors flow/scripts/infer.py of the official repo: config flow.configs.big_parts_strict_pvae, bf16,
50 steps, cfg 7.0, grid_res 384, no decimation (num_faces=-1), rembg (default u2net session) for
background removal when the input has no alpha, recenter with border 0.1, resize 518 (INTER_LINEAR),
white background. Seeding: kiui.seed_everything(seed) right before sampling (official uses seed+repeat_i).

What a "part" is here: PartPacker decodes two volumes (vol0, vol1); each is post-processed
(clean_mesh) and split into connected components; components with <=10 faces are dropped.
Each remaining component is one part, exactly as the official script exports `*_partJ.glb`.
The two raw volumes are saved as dual_volumes/vol{0,1}.glb and the colored scene as merged.glb.
Vertices are exported in the official GLB frame (after TRIMESH_GLB_EXPORT), no rescaling.

Run with PartPacker's own venv:
    CUDA_VISIBLE_DEVICES=4 flock /tmp/ppbench_gpu.lock \
      third_party/partpacker/.venv/bin/python ppbench/baselines/run_partpacker.py \
      --image IMG --out results/v2/<obj>/external --seeds 0 1 2
"""
import argparse, json, os, sys, time
from pathlib import Path

# kiui.seed_everything() seeds only the libraries it finds in the globals of the module that did `import kiui`
# (it walks the call stack). The official flow/scripts/infer.py has module-level `import kiui`, `import numpy as np`
# and `import torch` but no `import random`, so it seeds numpy + torch (+cuda) and not python `random`.
# These three module-level imports reproduce exactly that; do not add `import random` here.
import numpy as np
import torch
import kiui

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("PARTPACKER_REPO", ROOT / "third_party/partpacker/repo"))
WEIGHTS = ("nvidia/PartPacker", "17046fe1729d05db99cb11d7cc6885d777fcfc73")
DINO = ("facebook/dinov2-giant", "611a9d42f2335e0f921f1e313ad3c1b7178d206d")


def git_rev(p):
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def bounds(meshes):
    import numpy as np
    vs = [m.vertices for m in meshes if len(m.vertices)]
    if not vs:
        return None
    v = np.concatenate(vs)
    lo, hi = v.min(0), v.max(0)
    return {"min": lo.round(4).tolist(), "max": hi.round(4).tolist(), "extent": (hi - lo).round(4).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="parent dir; each run goes to <out>/<tag>_s<seed>/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--grid_res", type=int, default=384)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=7.0)
    ap.add_argument("--num_faces", type=int, default=-1, help="decimation target; -1 = none (official default)")
    ap.add_argument("--tag", default="partpacker")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))
    os.environ.setdefault("U2NET_HOME", os.path.expanduser(os.environ.get("U2NET_HOME", "~/.u2net")))
    sys.path.insert(0, str(REPO))

    import importlib
    import cv2, kiui, numpy as np, rembg, torch, trimesh
    from huggingface_hub import hf_hub_download, snapshot_download
    from flow.model import Model
    from flow.utils import get_random_color, recenter_foreground
    from vae.utils import postprocess_mesh

    TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)
    flow_ckpt = hf_hub_download(WEIGHTS[0], "flow.pt", revision=WEIGHTS[1])
    vae_ckpt = hf_hub_download(WEIGHTS[0], "vae.pt", revision=WEIGHTS[1])
    dino_dir = snapshot_download(DINO[0], revision=DINO[1], allow_patterns=["*.json", "model.safetensors"])
    # flow/model.py calls Dinov2Model.from_pretrained("facebook/dinov2-giant") (branch main); point it at the
    # pinned snapshot so the image encoder revision is fixed and HF_HUB_OFFLINE works. Weights are identical.
    import flow.model as _fm
    _orig_from_pretrained = _fm.Dinov2Model.from_pretrained

    class _PinnedDinov2:
        @staticmethod
        def from_pretrained(name, *args, **kw):
            return _orig_from_pretrained(dino_dir if name == DINO[0] else name, *args, **kw)

    _fm.Dinov2Model = _PinnedDinov2

    bg_remover = rembg.new_session()  # default model name: u2net

    def preprocess_image(path):
        input_image = kiui.read_image(path, mode="uint8", order="RGBA")
        had_alpha = input_image.shape[-1] == 4
        if not had_alpha:
            input_image = rembg.remove(input_image, session=bg_remover)
        mask = input_image[..., -1] > 0
        image = recenter_foreground(input_image, mask, border_ratio=0.1)
        image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        return image, had_alpha

    cfg = importlib.import_module("flow.configs.big_parts_strict_pvae").make_config()
    cfg.vae_ckpt_path = vae_ckpt
    ckpt = torch.load(flow_ckpt, weights_only=True)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model = Model(cfg).eval().cuda().bfloat16()
    model.load_state_dict(ckpt, strict=True)
    del ckpt

    out_root = Path(a.out)
    for seed in a.seeds:
        run = out_root / f"{a.tag}_s{seed}"
        (run / "parts").mkdir(parents=True, exist_ok=True)
        (run / "dual_volumes").mkdir(exist_ok=True)
        warnings = [
            "parts are connected components of the two decoded volumes (official export), not semantic parts; "
            "one semantic part may be split into several components and components with <=10 faces are dropped"
        ]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        image, had_alpha = preprocess_image(a.image)
        kiui.write_image(str(run / "input_processed.png"), image)
        x = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float().cuda()
        t1 = time.time()
        kiui.seed_everything(seed)
        with torch.inference_mode():
            results = model({"cond_images": x}, num_steps=a.num_steps, cfg_scale=a.cfg_scale)
        latent = results["latent"]
        vols, parts = [], []
        for k, sl in enumerate([slice(None, cfg.latent_size), slice(cfg.latent_size, None)]):
            with torch.inference_mode():
                r = model.vae({"latent": latent[:, sl, :]}, resolution=a.grid_res)
            v, f = r["meshes"][0]
            m = trimesh.Trimesh(v, f)
            m.vertices = m.vertices @ TRIMESH_GLB_EXPORT.T
            m = postprocess_mesh(m, a.num_faces)
            vols.append(m)
            m.export(run / "dual_volumes" / f"vol{k}.glb")
            for p in m.split(only_watertight=False):
                parts.append((k, p))
        n_raw = len(parts)
        parts = [(k, p) for k, p in parts if len(p.faces) > 10]
        wall = time.time() - t0
        if n_raw != len(parts):
            warnings.append(f"{n_raw - len(parts)} components with <=10 faces dropped (official filter)")
        parts_meta = []
        for j, (k, p) in enumerate(parts):
            p.visual.vertex_colors = get_random_color(j, use_float=True)
            p.export(run / "parts" / f"part_{j:02d}.glb")
            parts_meta.append({"index": j, "file": f"parts/part_{j:02d}.glb", "volume": k, "n_vertices": int(len(p.vertices)),
                               "n_faces": int(len(p.faces)), "watertight": bool(p.is_watertight)})
        if parts:
            trimesh.Scene([p for _, p in parts]).export(run / "merged.glb")
        else:
            warnings.append("no parts survived")
        meta = {
            "system": a.tag,
            "model": "PartPacker",
            "code_repo": "https://github.com/NVlabs/PartPacker", "code_commit": git_rev(REPO),
            "weights": {"repo": WEIGHTS[0], "revision": WEIGHTS[1], "files": ["flow.pt", "vae.pt"],
                        "image_encoder": {"repo": DINO[0], "revision": DINO[1],
                                          "note": "flow/model.py's Dinov2Model.from_pretrained('facebook/dinov2-giant') "
                                                  "redirected to this pinned snapshot (official code loads branch main)"}},
            "bg_removal": {"method": "rembg 2.0.60, rembg.new_session() default model 'u2net' (onnxruntime CPU), "
                                     "applied only if input has no alpha; then recenter_foreground(border_ratio=0.1), "
                                     "resize 518x518 INTER_LINEAR, composite on white (flow/scripts/infer.py)",
                           "u2net_onnx_md5": "60024c5c889badc19c04ad937298a77b (rembg release v0.0.0 u2net.onnx)",
                           "applied": not had_alpha},
            "input_image": str(Path(a.image).resolve()),
            "seed": seed,
            "seeding": "kiui.seed_everything(seed) immediately before sampling, as official infer.py with num_repeats=1; with the official module-level imports this seeds numpy, torch and torch.cuda (python random is not seeded)",
            "settings": {"config": "flow.configs.big_parts_strict_pvae", "dtype": "bfloat16", "num_steps": a.num_steps,
                         "cfg_scale": a.cfg_scale, "grid_res": a.grid_res, "vae_decode_mode": "hierarchical (default)",
                         "num_faces": a.num_faces, "component_min_faces": 11},
            "wall_s": round(wall, 1), "wall_s_generation_only": round(time.time() - t1, 1),
            "n_parts": len(parts), "n_parts_per_volume": [sum(1 for k, _ in parts if k == 0), sum(1 for k, _ in parts if k == 1)],
            "parts": parts_meta,
            "merged_mesh": "merged.glb" if parts else None,
            "dual_volumes": ["dual_volumes/vol0.glb", "dual_volumes/vol1.glb"],
            "output_convention": {
                "frame": "PartPacker VAE space [-1,1]^3 (object normalized), then permuted by the official "
                         "TRIMESH_GLB_EXPORT matrix [[0,1,0],[0,0,1],[1,0,0]] (new = (y, z, x)) for GLB export; all parts share one frame",
                "units": "unitless (normalized), no metric scale",
                "up_axis": "Y-up (glTF) after the official export permutation. Empirical check in 'bbox'.",
            },
            "bbox": bounds([p for _, p in parts]),
            "peak_gpu_mem_gb": {"torch_max_allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                "torch_max_reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)},
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
            "warnings": warnings,
        }
        (run / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: meta[k] for k in ("system", "seed", "n_parts", "wall_s", "peak_gpu_mem_gb", "bbox")}), flush=True)


if __name__ == "__main__":
    main()
