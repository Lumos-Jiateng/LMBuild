#!/usr/bin/env python3
"""BrickGPT: caption -> brick list -> LDraw. Standalone; imports nothing from ppbench.

Runs under the base interpreter with BrickGPT's source and the vendored Gurobi
wheel on the path. `use_gurobi=False` because the size-limited pip licence
rejects anything past two bricks, so BrickGPT falls back to its documented
connectivity-based stability rule. That is recorded in the meta file, not hidden.

    python ppbench/baselines/run_brickgpt.py --caption "..." --out DIR --seeds 0 1 2 --gpu 5
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRICKFORGE = Path(os.environ.get("PPBENCH_BRICKFORGE", str(Path(__file__).resolve().parents[2] / "third_party" / "BrickForge")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_bricks", type=int, default=2000, help="BrickGPT's own default; the three jeep runs passed 80, which did not bind (111, 112 and 201 bricks came out)")
    ap.add_argument("--tag", default="brickgpt")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    os.environ.setdefault("HF_HOME", os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(ROOT / ".deps"))
    sys.path.insert(0, str(BRICKFORGE / "third_party/BrickGPT/src"))
    import transformers
    from brickgpt.models import BrickGPT, BrickGPTConfig
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = BrickGPT(BrickGPTConfig(use_gurobi=False, device="cuda", max_bricks=a.max_bricks))
    for seed in a.seeds:
        transformers.set_seed(seed)
        t0 = time.time(); r = m(a.caption); dt = time.time() - t0
        stem = f"{a.tag}_s{seed}"
        (out / f"{stem}.ldr").write_text(r["bricks"].to_ldr())
        (out / f"{stem}.txt").write_text(r["bricks"].to_txt())
        meta = {"system": "brickgpt", "model": "AvaLovelace/BrickGPT (Llama-3.2-1B)", "caption": a.caption,
                "seed": seed, "n_bricks": len(r["bricks"]), "wall_s": round(dt, 1),
                "rejections": dict(r["rejection_reasons"]), "n_regenerations": r["n_regenerations"],
                "use_gurobi": False, "note": "stability rejection is connectivity-based, not Gurobi physics"}
        (out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
