"""Typical dimensions of the whole object, quoted from its (pinned) Wikipedia article.

Candidate sentences are those of the object article that contain a length with a unit. A non-contestant
model is asked which of them state a typical dimension of the whole object (not a record, a historical
object, or a part); every answer is then checked mechanically against the verbatim sentence: the numbers
it reports must occur in that sentence together with the unit, or the answer is dropped. What survives
widens the size band real instances give (afford/calib.py) and is listed, with its quote, in the report.

    python -m ppbench.v2.afford.wiki [--tasks all] [--model qwen2.5-vl-32b]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ppbench.v2.afford import core
from ppbench.v2.task import RESULTS, ROOT

UNIT_M = {"m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0, "cm": 0.01, "centimetres": 0.01,
          "centimeters": 0.01, "mm": 0.001, "millimetres": 0.001, "millimeters": 0.001, "in": 0.0254,
          "inch": 0.0254, "inches": 0.0254, "ft": 0.3048, "foot": 0.3048, "feet": 0.3048}
NUM_UNIT = re.compile(r"\d+(?:[.,]\d+)?\s*(?:–|-|to)?\s*\d*(?:[.,]\d+)?\s*(mm|cm|m|metres?|meters?|centimet(?:re|er)s|"
                      r"millimet(?:re|er)s|in|inch(?:es)?|ft|feet|foot)\b")


def article(tid):
    from ppbench.v2.task import task_dir
    for f in sorted((task_dir(tid) / "knowledge").glob("sources*.json")):
        if f.exists():
            return json.loads(f.read_text())["wikipedia"]["object"]
    return None


def candidates(text):
    sents = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [s.strip() for s in sents if NUM_UNIT.search(s) and 20 < len(s) < 600][:80]


def _num_in(x, s):
    for form in {f"{x:g}", f"{x:.1f}", f"{x:.2f}", f"{int(x)}" if float(x).is_integer() else f"{x:g}"}:
        if re.search(rf"(?<![\d.]){re.escape(form)}(?![\d])", s):
            return True
    return False


def extract(tid, client, model):
    t = core.task(tid)
    art = article(tid)
    out = {"task": tid, "article": None, "facts": [], "candidates": 0}
    if not art:
        return out
    out["article"] = {"title": art["title"], "revision": art["revision"], "url": art["url"]}
    cands = candidates(art.get("text"))
    out["candidates"] = len(cands)
    if not cands:
        return out
    name = t.raw["object"]["name"]
    listing = "\n".join(f"[{i}] {s}" for i, s in enumerate(cands))
    prompt = (f"Below are sentences from the Wikipedia article '{art['title']}'. Find the sentences that state a "
              f"TYPICAL, standard or common overall dimension of a whole {name} as it is made today (overall height, "
              f"length, width, depth or diameter of the complete object). Ignore records, historical or unusual "
              f"examples, dimensions of parts, ranges of motion, distances and anything that is not the size of the "
              f"object itself.\n\n{listing}\n\nAnswer with JSON only: a list of objects "
              f'{{"sentence": <number>, "dimension": "height|length|width|depth|diameter", "min": <number>, '
              f'"max": <number>, "unit": "<unit exactly as written in the sentence>"}}. Answer [] if there is none.')
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                       max_tokens=600, temperature=0.0)
    txt = r.choices[0].message.content or ""
    m = re.search(r"\[.*\]", txt, re.S)
    try:
        facts = json.loads(m.group(0)) if m else []
    except ValueError:
        facts = []
    for f in facts if isinstance(facts, list) else []:
        try:
            s = cands[int(f["sentence"])]
            lo, hi = float(f["min"]), float(f.get("max") or f["min"])
            unit = str(f["unit"]).strip().lower().rstrip(".")
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        ok = unit in UNIT_M and re.search(rf"\b{re.escape(unit)}\b", s.lower()) and _num_in(lo, s) and _num_in(hi, s)
        rec = {"quote": s, "dimension": f.get("dimension"), "min": lo, "max": hi, "unit": unit, "verified": bool(ok)}
        if ok:
            rec["min_m"], rec["max_m"] = sorted((lo * UNIT_M[unit], hi * UNIT_M[unit]))
        out["facts"].append(rec)
    out["model"] = model
    return out


def sizes(tid):
    """Verified object dimensions mapped onto the envelope's axes (height / long / wide)."""
    f = RESULTS / tid / "afford" / "wiki_sizes.json"
    if not f.exists():
        return []
    w = json.loads(f.read_text())
    out = []
    for x in w.get("facts", []):
        if not x.get("verified"):
            continue
        dim = (x.get("dimension") or "").lower()
        target = "height" if dim == "height" else ("horiz" if dim in ("length", "width", "depth", "diameter") else None)
        if target:
            out.append({"target": "object", "dim": target, "min_m": x["min_m"], "max_m": x["max_m"],
                        "quote": x["quote"], "source": (w.get("article") or {}).get("url")})
    return out


def main(argv=None):
    from openai import OpenAI
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--model", default="qwen2.5-vl-32b")
    ap.add_argument("--base-url", default="http://127.0.0.1:8411/v1")
    ns = ap.parse_args(argv)
    client = OpenAI(base_url=ns.base_url, api_key="x", timeout=300)
    tids = core.task_ids() if ns.tasks == "all" else ns.tasks.split(",")
    for tid in tids:
        t = core.task(tid)
        if core.medium(t) != "real":
            continue
        w = extract(tid, client, ns.model)
        p = RESULTS / tid / "afford" / "wiki_sizes.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(w, indent=1))
        v = [f for f in w["facts"] if f["verified"]]
        print(f"{tid:26s} cands={w['candidates']:3d} facts={len(w['facts'])} verified={len(v)} "
              + "; ".join(f"{f['dimension']} {f['min_m']:.2f}-{f['max_m']:.2f} m" for f in v)[:200], flush=True)


if __name__ == "__main__":
    main()
