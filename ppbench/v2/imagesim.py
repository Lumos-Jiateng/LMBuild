"""Image consistency: DINOv2-large cosine similarity between a design's render and
the task's condition image (and, as the upper anchor, the reference render from
the same camera). The same model and preprocessing as the benchmark's own image
fidelity check (`reference_data/benchmark_v2/images/fidelity.py`), run on the CPU.

Runs in `.venv_imagegen` (torch + transformers):
    .venv_imagegen/bin/python -m ppbench.v2.imagesim pairs.json out.json
pairs.json: {"condition": png, "items": {key: png}}
"""
from __future__ import annotations

import json
import sys


def main(pairs_path, out_path):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    torch.set_num_threads(16)
    job = json.load(open(pairs_path))
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").eval()

    def emb(path):
        im = Image.open(path).convert("RGB")
        with torch.no_grad():
            out = model(**proc(images=im, return_tensors="pt"))
        e = out.pooler_output[0]
        return e / e.norm()

    c = emb(job["condition"])
    res = {k: float(emb(p) @ c) for k, p in job["items"].items()}
    json.dump({"model": "facebook/dinov2-large", "condition": job["condition"], "cosine": res}, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
