"""Download LMBuild data from Hugging Face (datasets/Lumos-Jiateng/LMBuild) into the places the code expects.

    python scripts/download_data.py                      # core benchmark (200 objects) + evaluator fixtures  (~12 GB)
    python scripts/download_data.py --full               # + LMBuild-Full (2,549 objects)                      (+86 GB)
    python scripts/download_data.py --designs            # + all 6,858 scored designs of the 30 systems         (+80 GB)
    python scripts/download_data.py --demos              # + example trajectories, final designs, GIFs, human-study renders
    python scripts/download_data.py --all                # everything

Hugging Face layout -> local layout (relative to the repository root):
    core/                              -> benchmark/            (core_v2.json, tasks/, assets/)
    full/                              -> benchmark/            (merged with core; use PPBENCH_CORE=benchmark/full_v2.json)
    results/fixtures/                  -> results/v2/           (task snapshots, anchors, Level 2 sheets, judge answers)
    results/designs/                   -> designs/              (stage them with scripts/stage_designs.py, then evaluate.sh)
    results/demos/                     -> demos/
    results/human_evaluation_renders/  -> human_evaluation/renders/
Files are downloaded into the Hugging Face cache and hard-linked (or copied) into place; re-running resumes.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Lumos-Jiateng/LMBuild"
PARTS = {  # name: (allow pattern in the HF repo, HF prefix, local destination)
    "core": ("core/**", "core", "benchmark"),
    "fixtures": ("results/fixtures/**", "results/fixtures", "results/v2"),
    "full": ("full/**", "full", "benchmark"),
    "designs": ("results/designs/**", "results/designs", "designs"),
    "demos": ("results/demos/**", "results/demos", "demos"),
    "renders": ("results/human_evaluation_renders/**", "results/human_evaluation_renders", "human_evaluation/renders"),
}


def place(src: Path, dst: Path) -> int:
    n = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        q = dst / p.relative_to(src)
        if q.exists():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        real = p.resolve()
        try:
            os.link(real, q)
        except OSError:
            shutil.copy2(real, q)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("full", "designs", "demos", "all"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--cache", default=None, help="download directory (default: the Hugging Face cache)")
    ns = ap.parse_args()
    from huggingface_hub import snapshot_download

    want = ["core", "fixtures"]
    if ns.full or ns.all:
        want.append("full")
    if ns.designs or ns.all:
        want.append("designs")
    if ns.demos or ns.all:
        want += ["demos", "renders"]
    for k in want:
        pattern, prefix, dest = PARTS[k]
        print(f"[{k}] downloading {pattern} ...", flush=True)
        snap = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[pattern], cache_dir=ns.cache,
                                      max_workers=16))
        n = place(snap / prefix, ROOT / dest)
        print(f"[{k}] {n} new files -> {dest}/", flush=True)
    print("done. Next: bash scripts/setup.sh  (environments + smoke test)")


if __name__ == "__main__":
    main()
