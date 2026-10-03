#!/usr/bin/env python3
"""PartCrafter (wgsxm/PartCrafter): one RGB image -> N part meshes. Standalone; imports nothing from ppbench.

Mirrors scripts/inference_partcrafter.py of the official repo (fp16, 50 steps, cfg 7.0,
1024 tokens/part, RMBG-1.4 background removal via `--rmbg` as the README asks for custom
images). Differences from the official script, all bookkeeping only:
  * weights come from the HF cache at pinned revisions instead of repo/pretrained_weights/;
  * a part whose decode fails (the pipeline's bare `except` returns None) is NOT replaced by
    the official one-vertex dummy mesh; its index is listed under meta["failed_part_indices"];
  * no --render, no VLM part suggestion, no Gemini style transfer.
Output meshes are exported exactly as the pipeline returns them (no rescaling or re-orientation).

Run with PartCrafter's own venv:
    CUDA_VISIBLE_DEVICES=4 flock /tmp/ppbench_gpu.lock \
      third_party/partcrafter/.venv/bin/python ppbench/baselines/run_partcrafter.py \
      --image IMG --out results/v2/<obj>/external --num_parts 15 --seeds 0 1 2
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("PARTCRAFTER_REPO", ROOT / "third_party/partcrafter/repo"))
WEIGHTS = ("wgsxm/PartCrafter", "69a0ffc1dad5e48e7e5ed91c0609f2b1276eb31f")
RMBG = ("briaai/RMBG-1.4", "2ceba5a5efaec153162aedea169f76caf9b46cf8")


def git_rev(p):
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def bounds(meshes):
    import numpy as np
    vs = [m.vertices for m in meshes if m is not None and len(m.vertices)]
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
    ap.add_argument("--num_parts", type=int, default=15)
    ap.add_argument("--num_tokens", type=int, default=1024)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.0)
    ap.add_argument("--max_num_expanded_coords", type=int, default=int(1e9))
    ap.add_argument("--no_rmbg", action="store_true", help="skip RMBG-1.4 (only for images that already have alpha)")
    ap.add_argument("--tag", default=None, help="default: partcrafter_np<num_parts>")
    ap.add_argument("--num_parts_source", default="", help="free text recorded in meta: why this part count")
    a = ap.parse_args()
    assert 1 <= a.num_parts <= 16, "PartCrafter MAX_NUM_PARTS is 16"
    os.environ.setdefault("HF_HOME", os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))
    sys.path.insert(0, str(REPO))

    import numpy as np, torch, trimesh
    from huggingface_hub import snapshot_download
    from accelerate.utils import set_seed
    from src.utils.data_utils import get_colored_mesh_composition
    from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
    from src.utils.image_utils import prepare_image
    from src.models.briarmbg import BriaRMBG

    device, dtype = "cuda", torch.float16
    pc_dir = snapshot_download(repo_id=WEIGHTS[0], revision=WEIGHTS[1])
    rmbg_dir = snapshot_download(repo_id=RMBG[0], revision=RMBG[1])
    rmbg_net = BriaRMBG.from_pretrained(rmbg_dir).to(device).eval()
    pipe = PartCrafterPipeline.from_pretrained(pc_dir).to(device, dtype)

    tag = a.tag or f"partcrafter_np{a.num_parts}"
    out_root = Path(a.out)
    for seed in a.seeds:
        run = out_root / f"{tag}_s{seed}"
        (run / "parts").mkdir(parents=True, exist_ok=True)
        warnings = []
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        set_seed(seed)
        with torch.no_grad():
            if a.no_rmbg:
                from PIL import Image
                img = Image.open(a.image)
            else:
                img = prepare_image(a.image, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
            t1 = time.time()
            meshes = pipe(
                image=[img] * a.num_parts,
                attention_kwargs={"num_parts": a.num_parts},
                num_tokens=a.num_tokens,
                generator=torch.Generator(device=pipe.device).manual_seed(seed),
                num_inference_steps=a.num_inference_steps,
                guidance_scale=a.guidance_scale,
                max_num_expanded_coords=a.max_num_expanded_coords,
                use_flash_decoder=False,
            ).meshes
        wall = time.time() - t0
        img.save(run / "input_processed.png")
        failed = [i for i, m in enumerate(meshes) if m is None or len(m.faces) == 0]
        if failed:
            warnings.append(f"decode failed (pipeline returned None) for part slots {failed}; no file written for them")
        parts_meta = []
        for i, m in enumerate(meshes):
            if i in failed:
                continue
            m.export(run / "parts" / f"part_{i:02d}.glb")
            parts_meta.append({"index": i, "file": f"parts/part_{i:02d}.glb", "n_vertices": int(len(m.vertices)),
                               "n_faces": int(len(m.faces)), "watertight": bool(m.is_watertight)})
        good = [m for i, m in enumerate(meshes) if i not in failed]
        if good:
            get_colored_mesh_composition(good).export(run / "merged.glb")
        meta = {
            "system": tag,
            "model": "PartCrafter",
            "code_repo": "https://github.com/wgsxm/PartCrafter", "code_commit": git_rev(REPO),
            "weights": {"repo": WEIGHTS[0], "revision": WEIGHTS[1]},
            "bg_removal": {"method": "BriaRMBG (briaai/RMBG-1.4) via src.utils.image_utils.prepare_image: OTSU-thresholded mask, "
                                     "small components removed, composited on white, cropped to bbox with 10% padding",
                           "weights_revision": RMBG[1], "enabled": not a.no_rmbg},
            "input_image": str(Path(a.image).resolve()),
            "seed": seed,
            "seeding": "accelerate.utils.set_seed(seed) + torch.Generator('cuda').manual_seed(seed) (as official script)",
            "settings": {"num_parts": a.num_parts, "num_parts_source": a.num_parts_source, "num_tokens": a.num_tokens,
                         "num_inference_steps": a.num_inference_steps, "guidance_scale": a.guidance_scale,
                         "dtype": "float16", "max_num_expanded_coords": a.max_num_expanded_coords,
                         "bounds": [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005], "dense_octree_depth": 8,
                         "hierarchical_octree_depth": 9},
            "wall_s": round(wall, 1), "wall_s_generation_only": round(time.time() - t1, 1),
            "n_parts_requested": a.num_parts, "n_parts": len(good), "failed_part_indices": failed,
            "parts": parts_meta,
            "merged_mesh": "merged.glb" if good else None,
            "output_convention": {
                "frame": "TripoSG/PartCrafter normalized object space; all parts share one frame (no per-part transform); "
                         "decoded inside bounds [-1.005, 1.005]^3; vertices written verbatim to GLB by trimesh",
                "units": "unitless (normalized), no metric scale",
                "up_axis": "Y-up (glTF convention; TripoSG training meshes are glTF/Objaverse normalized). Empirical check in 'bbox'.",
            },
            "bbox": bounds(good),
            "peak_gpu_mem_gb": {"torch_max_allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                "torch_max_reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)},
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
            "warnings": warnings,
        }
        (run / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: meta[k] for k in ("system", "seed", "n_parts", "wall_s", "peak_gpu_mem_gb", "bbox", "warnings")}), flush=True)


if __name__ == "__main__":
    main()
