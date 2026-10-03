#!/usr/bin/env python3
"""LegoACE text-conditioned inference: caption -> LDraw. Standalone.

Runs under the base interpreter with LegoACE's source and its vendored
dependency directory *first* on the path: that directory carries the
transformers 4.49 LegoACE was written against, and a newer transformers picks
up the vendored `regex` and fails on a circular import. Mirrors
BrickForge/runs/t2lego/legoace_gen.py, which produced the val128 set.

    python ppbench/baselines/run_legoace.py --caption "..." --out DIR --seeds 0 1 2 --gpu 5
"""
import argparse, json, os, sys, time
from pathlib import Path

BRICKFORGE = Path(os.environ.get("PPBENCH_BRICKFORGE", str(Path(__file__).resolve().parents[2] / "third_party" / "BrickForge")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tag", default="legoace")
    ap.add_argument("--max_length", type=int, default=5000)
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    os.environ.setdefault("HF_HOME", os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("LEGOACE_DATA_ROOT", str(BRICKFORGE / "third_party/data/legoace"))
    out = Path(a.out).resolve()            # before chdir: LegoACE's dataset code needs its own cwd
    src = BRICKFORGE / "third_party/LegoACE"
    sys.path.insert(0, str(BRICKFORGE / "third_party/legoace_deps")); sys.path.insert(0, str(src))
    os.chdir(src)
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    from dataset.textDataset import TextDataset
    from model.llama_text_condition import TextConditionModel
    from model.logitsprocessor import DynamicRangeMaskingProcessor
    ds = TextDataset("bricklink-text", "val", pos_range=1280)
    lp = DynamicRangeMaskingProcessor(ds.position_range, ds.num_rotations, ds.num_classes)
    eos = ds.get_vocab_size() - 1
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    model = TextConditionModel.from_pretrained("VAST-AI/LegoACE", subfolder="text").to("cuda").eval()
    out.mkdir(parents=True, exist_ok=True)
    # LegoACE conditions on CLIP ViT-B/32 text features, whose position table is 77 tokens. The attributes and
    # functional prompts are longer (e.g. 181 tokens), so they are cut to the encoder's 77, as CLIP always does;
    # the name_only captions all fit, so their outputs are unchanged (2026-09-22).
    n_tok = len(tok(a.caption)["input_ids"])
    enc = tok([a.caption], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
    enc = {k: v.to("cuda") for k, v in enc.items()}
    for seed in a.seeds:
        torch.manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            cond = clip(**enc)[0]
            ids = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
            am = torch.ones((1, 1), dtype=torch.bool, device="cuda")
            gen = model.generate(input_ids=ids, use_cache=True, condition_embeds=cond, pad_token_id=eos,
                                 bos_token_id=0, eos_token_id=eos, max_length=a.max_length + 2,
                                 attention_mask=am, logits_processor=[lp], do_sample=True, top_k=10, top_p=0.95)
            gen[:, -1] = eos
        ldr = ds.convert_npy_to_ldr(gen[0].cpu().numpy())
        stem = f"{a.tag}_s{seed}"
        (out / f"{stem}.ldr").write_text("".join(ldr))
        meta = {"system": "legoace", "model": "VAST-AI/LegoACE text", "caption": a.caption, "seed": seed,
                "caption_tokens": n_tok, "caption_truncated_to": tok.model_max_length if n_tok > tok.model_max_length else None,
                "n_bricks": len(ldr), "wall_s": round(time.time() - t0, 1), "top_k": 10, "top_p": 0.95}
        (out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
