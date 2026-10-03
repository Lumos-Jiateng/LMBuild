#!/usr/bin/env python3
"""BrickNet: caption -> connector-graph text -> LDraw. Standalone.

Two steps, each in the interpreter that can run it. Generation uses BrickNet's
own `scripts/generate.py` (base Qwen3 plus the PT and SFT LoRA adapters) under
a peft-capable env. Decoding the text to poses uses `python -m bricknet path2ldr`,
which is BrickNet's own decoder, because the connector algebra lives there.

    python ppbench/baselines/run_bricknet.py --caption "..." --out DIR --size 0.6b --n 3 --gpu 6
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRICKFORGE = Path(os.environ.get("PPBENCH_BRICKFORGE", str(Path(__file__).resolve().parents[2] / "third_party" / "BrickForge")))
GEN_PYTHON = os.environ.get("PPBENCH_PEFT_PYTHON",
                            "python")
SIZES = {"0.6b": ("Qwen/Qwen3-0.6B", "kulits/BrickNet-0.6B-PT", "kulits/BrickNet-0.6B-SFT"),
         "14b": ("Qwen/Qwen3-14B", "kulits/BrickNet-14B-PT", "kulits/BrickNet-14B-SFT")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="0.6b", choices=sorted(SIZES))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    tag = f"bricknet-{a.size}"
    base, pt, sft = SIZES[a.size]
    prompts = out / f"{tag}.prompts.jsonl"
    prompts.write_text(json.dumps({"id": 0, "caption": a.caption}) + "\n")
    gen = out / f"{tag}.gen.jsonl"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, HF_HUB_OFFLINE="1",
               HF_HOME=os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))
    t0 = time.time()
    cmd = [GEN_PYTHON, str(BRICKFORGE / "third_party/BrickNet/scripts/generate.py"),
           "--model", base, "--lora", pt, "--lora", sft, "--output", str(gen),
           "--prompts_file", str(prompts), "--n_per_prompt", str(a.n), "--batch_size", str(a.n),
           "--seed", str(a.seed)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(BRICKFORGE / "third_party/BrickNet"))
    # generate.py merges its per-process shards into --output when it finishes;
    # an interrupted run leaves the shards, so read whichever exists.
    if gen.exists() and gen.stat().st_size > 0:
        rows = [json.loads(l) for l in open(gen)]
    else:
        shards = sorted(out.glob(f"{tag}.gen.jsonl.shard*"))
        rows = [json.loads(l) for s in shards for l in open(s)]
    with open(gen, "w") as fh:
        for i, r in enumerate(rows):
            r["id"] = f"{tag}_s{a.seed + i}"
            fh.write(json.dumps(r) + "\n")
    env2 = dict(os.environ, BRICKNET_DATA=str(BRICKFORGE / "third_party/data/bricknet"),
                PYTHONPATH=f"{BRICKFORGE / 'third_party/BrickNet/src'}:{ROOT / '.deps'}")
    subprocess.run([sys.executable, "-m", "bricknet", "path2ldr", str(gen), "-o", str(out)], check=True, env=env2)
    for i, r in enumerate(rows):
        meta = {"system": tag, "model": f"{base} + {pt} + {sft}", "caption": a.caption, "seed": a.seed + i,
                "wall_s_total": round(time.time() - t0, 1), "decoded": (out / f"{r['id']}.ldr").exists()}
        (out / f"{r['id']}.meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
