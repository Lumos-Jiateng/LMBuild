"""Adjudicated half of 3.2 (aesthetics) and the relevance gate, by a VLM.

Judged on renders, never on part lists. Each design is scored on its own (decided with the benchmark
author on 2026-09-14, replacing an earlier pairwise design): the judge sees the condition photo and one
design's clay renders, and gives 1-10 on four criteria with written anchors. Nothing tells it which system
made the design. Two judges that are not contestants score every design independently; their rank
agreement is reported next to the scores, and the reference object is scored like any design, as a
calibration point. A human study is still owed.

Images per request are capped at two (the shared vLLM servers enforce it): the condition photo, and the
design's four clay views tiled into one image. Renders are neutral clay so a palette cannot sway the judge.

    .venv_eval/bin/python -m ppbench.v2.judge rubric --items items.json --out judge_<model>.json \
        --condition photo.png --model <served name> --base-url http://127.0.0.1:8101/v1
    ("pairs" is kept as an alias of "rubric" so already-running scripts call the per-design scoring.)
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import re
import time
from io import BytesIO
from pathlib import Path

CRITERIA = {
    "resemblance": "Which design better matches the object in the photograph: overall shape, the parts it has and how they are arranged?",
    "proportion": "Which design has more realistic, deliberate proportions and symmetry for this kind of object?",
    "coherence": "Which design looks like one coherent, finished product: parts that fit together, a consistent style, nothing floating, misplaced or broken?",
    "overall": "Overall, which design would an experienced product designer prefer?",
}

PAIR_PROMPT = """You are judging 3D designs of an object. The first image is a photograph of the target object.
The second image shows two candidate designs rendered in plain grey clay from the same four views: design A on the
LEFT half, design B on the RIGHT half. Judge only what is visible; ignore rendering quality and colour.

Answer each question with "A", "B" or "tie":
{questions}

Reply with JSON only, e.g. {{"resemblance": "A", "proportion": "tie", "coherence": "B", "overall": "A", "reason": "one sentence"}}"""

GATE_PROMPT = """The image shows a 3D design rendered in plain grey clay from four views.
What object is it? Is it recognisably a {name}? Reply with JSON only:
{{"object": "<what you see>", "is_target": true or false, "confidence": <0 to 1>}}"""


def _b64(im):
    buf = BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def side_by_side(a_path, b_path, h=640):
    from PIL import Image, ImageDraw, ImageFont
    A = Image.open(a_path).convert("RGB")
    B = Image.open(b_path).convert("RGB")
    A = A.resize((int(A.width * h / A.height), h))
    B = B.resize((int(B.width * h / B.height), h))
    im = Image.new("RGB", (A.width + B.width + 16, h + 40), (255, 255, 255))
    im.paste(A, (0, 40))
    im.paste(B, (A.width + 16, 40))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except OSError:
        f = ImageFont.load_default()
    d.text((A.width // 2 - 60, 4), "design A", fill=(0, 0, 0), font=f)
    d.text((A.width + 16 + B.width // 2 - 60, 4), "design B", fill=(0, 0, 0), font=f)
    return im


def _ask(client, model, content, max_tokens=400, retries=3):
    for k in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}],
                                               max_tokens=max_tokens, temperature=0.0)
            txt = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                return json.loads(m.group(0)), txt
        except Exception as e:  # transient server errors are retried, then recorded
            txt = f"error: {e}"
            time.sleep(5)
    return None, txt



RUBRIC = {
    "resemblance": "How closely does the design match the object in the photograph: overall shape, the parts it has, and how they are arranged? 1 = a different kind of object; 4 = the right kind of object but major parts missing or misplaced; 7 = clearly this object with minor differences; 10 = matches the photograph in every visible respect.",
    "proportion": "Are the proportions and symmetry realistic and deliberate for this kind of object? 1 = grossly distorted; 4 = noticeably off (e.g. a part far too large or lopsided); 7 = plausible with small issues; 10 = convincing real-world proportions.",
    "coherence": "Does it look like one coherent, finished product: parts that meet and fit together, a consistent style, nothing floating, intersecting or broken? 1 = scattered or broken pieces; 4 = several visible gaps, floating or clashing parts; 7 = mostly clean with a few flaws; 10 = fully resolved.",
    "overall": "Overall design quality, as an experienced product designer would rate it. 1 = unusable; 4 = poor; 7 = good; 10 = excellent.",
}

RUBRIC_PROMPT = """You are rating a 3D design of an object. The first image is a photograph of the target object.
The second image shows the design rendered in plain grey clay from four views (condition camera, front-right,
front, right side). Judge only what is visible; ignore rendering quality and colour. Rate each criterion with
an integer from 1 to 10 using the anchors given:
{questions}

Reply with JSON only, e.g. {{"resemblance": 6, "proportion": 7, "coherence": 5, "overall": 6, "reason": "one sentence"}}"""


def run_rubric(items, cond_image, model, base_url, out, api_key="local", workers=16):
    """items: {key: clay grid png}. One call per design; scores normalised to [0, 1] as (s - 1) / 9."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)
    cond = _b64(Image.open(cond_image).convert("RGB").resize((640, 640)))
    qs = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())

    def one(kp):
        k, p = kp
        img = Image.open(p).convert("RGB")
        img.thumbnail((960, 960))
        content = [{"type": "text", "text": RUBRIC_PROMPT.format(questions=qs)},
                   {"type": "image_url", "image_url": {"url": cond}}, {"type": "image_url", "image_url": {"url": _b64(img)}}]
        ans, raw = _ask(client, model, content, max_tokens=300)
        scores = {}
        for c in RUBRIC:
            try:
                v = float((ans or {}).get(c))
                scores[c] = max(0.0, min(1.0, (v - 1) / 9))
            except (TypeError, ValueError):
                scores[c] = None
        return k, {"raw_scores": {c: (ans or {}).get(c) for c in RUBRIC}, "scores": scores,
                   "reason": (ans or {}).get("reason"), "raw": raw[:400]}

    with ThreadPoolExecutor(workers) as ex:
        per = dict(ex.map(one, sorted(items.items())))
    res = {"model": model, "design": "per-design rubric, 1-10 per criterion, normalised (s-1)/9",
           "rubric": RUBRIC, "n_items": len(items), "items": per,
           "n_unparsed": sum(1 for v in per.values() if v["scores"]["overall"] is None)}
    Path(out).write_text(json.dumps(res, indent=1))
    return res


def agreement(judge_files):
    """Spearman rank correlation of the 'overall' score between every pair of judges, on shared items."""
    from itertools import combinations
    import numpy as np
    js = {Path(f).stem: json.loads(Path(f).read_text()) for f in judge_files}
    out = {}
    for a, b in combinations(sorted(js), 2):
        keys = [k for k in js[a]["items"] if k in js[b]["items"]
                and js[a]["items"][k]["scores"]["overall"] is not None and js[b]["items"][k]["scores"]["overall"] is not None]
        if len(keys) < 3:
            continue
        x = np.array([js[a]["items"][k]["scores"]["overall"] for k in keys])
        y = np.array([js[b]["items"][k]["scores"]["overall"] for k in keys])
        rx, ry = x.argsort().argsort(), y.argsort().argsort()
        out[f"{a}|{b}"] = {"n": len(keys), "spearman_overall": float(np.corrcoef(rx, ry)[0, 1])}
    return out

def sample_pairs(keys, k=3, anchor="reference", seed=0):
    """Each design meets `k` random opponents plus the reference anchor, each pair in both orders.
    A connected comparison graph is all Bradley-Terry needs; all-pairs grows quadratically and on 100+
    designs costs tens of thousands of judge calls for little extra precision."""
    import random
    rng = random.Random(seed)
    keys = sorted(keys)
    pairs = set()
    for a in keys:
        others = [b for b in keys if b != a and b != anchor]
        opp = rng.sample(others, min(k, len(others)))
        if anchor in keys and a != anchor:
            opp.append(anchor)
        for b in opp:
            pairs.add(tuple(sorted((a, b))))
    return [(a, b) for x, y in sorted(pairs) for a, b in ((x, y), (y, x))]


def run_pairs(items, cond_image, model, base_url, out, api_key="local", workers=16, k=3):
    """items: {key: clay grid png}. Sampled pairs (see sample_pairs), both orders."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)
    cond = _b64(Image.open(cond_image).convert("RGB").resize((640, 640)))
    qs = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    jobs = sample_pairs(items, k=k) if len(items) > 12 else [(a, b) for a, b in itertools.permutations(sorted(items), 2)]

    def one(ab):
        a, b = ab
        comp = _b64(side_by_side(items[a], items[b]))
        content = [{"type": "text", "text": PAIR_PROMPT.format(questions=qs)},
                   {"type": "image_url", "image_url": {"url": cond}}, {"type": "image_url", "image_url": {"url": comp}}]
        ans, raw = _ask(client, model, content)
        return {"A": a, "B": b, "answer": ans, "raw": raw[:400]}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, jobs))
    res = {"model": model, "criteria": CRITERIA, "n_items": len(items), "n_calls": len(rows),
           "design": "all pairs" if len(items) <= 12 else f"{k} random opponents + reference anchor, both orders", "rows": rows}
    res["ratings"] = {c: bradley_terry(rows, c, sorted(items)) for c in CRITERIA}
    res["position_bias"] = position_bias(rows)
    Path(out).write_text(json.dumps(res, indent=1))
    return res


def run_gate(items, name, model, base_url, out, api_key="local", workers=16):
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)

    def one(kp):
        k, p = kp
        img = _b64(Image.open(p).convert("RGB"))
        ans, raw = _ask(client, model, [{"type": "text", "text": GATE_PROMPT.format(name=name)},
                                        {"type": "image_url", "image_url": {"url": img}}], max_tokens=200)
        return k, {"answer": ans, "raw": raw[:300]}
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, sorted(items.items())))
    Path(out).write_text(json.dumps({"model": model, "target": name, "items": res}, indent=1))
    return res


def bradley_terry(rows, crit, keys, iters=200):
    """Win counts (tie = half a win to each) -> BT strengths by the MM algorithm, scaled to mean 0 log-strength."""
    import numpy as np
    ix = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    W = np.zeros((n, n))
    for r in rows:
        v = (r["answer"] or {}).get(crit)
        a, b = ix[r["A"]], ix[r["B"]]
        if v == "A":
            W[a, b] += 1
        elif v == "B":
            W[b, a] += 1
        elif v == "tie":
            W[a, b] += 0.5
            W[b, a] += 0.5
    p = np.ones(n)
    N = W + W.T
    for _ in range(iters):
        wins = W.sum(1) + 0.1                         # a small prior keeps a never-winning item finite
        denom = np.array([sum(N[i, j] / (p[i] + p[j]) for j in range(n) if j != i) for i in range(n)]) + 0.2 / (p + 1)
        p = wins / denom
        p = p / np.exp(np.mean(np.log(p)))
    wr = {k: float(W[ix[k]].sum() / max(N[ix[k]].sum(), 1)) for k in keys}
    return {k: {"log_strength": float(np.log(p[ix[k]])), "win_rate": wr[k]} for k in keys}


def position_bias(rows):
    first = sum(1 for r in rows for c in CRITERIA if (r["answer"] or {}).get(c) == "A")
    second = sum(1 for r in rows for c in CRITERIA if (r["answer"] or {}).get(c) == "B")
    return {"picked_left": first, "picked_right": second,
            "left_rate": first / max(first + second, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rubric", "pairs", "gate"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--condition")
    ap.add_argument("--name", default="office chair")
    ns = ap.parse_args()
    items = json.loads(Path(ns.items).read_text())
    if ns.mode in ("pairs", "gate"):
        # superseded (2026-09-14): the adjudicated checks are ppbench.v2.vlm_checks, grounded in evidence rendered
        # from each design's 3D structure; already-running scripts that call these modes do nothing now
        print(json.dumps({"superseded_by": "ppbench.v2.vlm_checks", "mode": ns.mode}))
        return
    if ns.mode == "rubric":
        r = run_rubric(items, ns.condition, ns.model, ns.base_url, ns.out)
        print(json.dumps({"n_items": r["n_items"], "n_unparsed": r["n_unparsed"]}))



if __name__ == "__main__":
    main()
