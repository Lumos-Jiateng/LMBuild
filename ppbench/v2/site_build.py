"""Build the single-file results page from site/data.json.

Images are embedded as JPEG data URIs (the page is published as one file), so
this step downsizes: a 2x2 part-coloured view grid per design and one clay view.

    .venv_eval/bin/python -m ppbench.v2.site_build --task swivel_office_chair
"""
from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

from ppbench.v2.task import RESULTS, ROOT

TEMPLATE = Path(__file__).with_name("site_template.html")


def jpeg(path, side=640, q=80):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def slim(rec):
    """Only what the page shows."""
    D = rec["dims"]
    keep = {}
    for k, v in D.items():
        m = dict(v["metrics"])
        for big in ("per_part", "sweeps", "colliding_pairs", "floating_parts", "unrecognised_roles", "best"):
            if big in m and isinstance(m[big], list):
                m[big] = m[big][:12]
        keep[k] = {**v, "metrics": m}
    meta = rec.get("meta") or {}
    return {"system": rec["system"], "tier": rec["tier"], "item": rec["item"], "notes": rec.get("notes", []),
            "dims": keep, "attributes": rec.get("attributes", []), "headline": rec.get("headline", {}),
            "meta": {k: meta.get(k) for k in ("protocol", "counters", "round_counters", "created_parts", "scale_mode",
                                              "oracle_scale", "oracle_yaw_deg", "transport", "wall_s", "family", "input",
                                              "submitted_final", "loop")},
            "eval_wall_s": rec.get("eval_wall_s")}


def build(task_id):
    base = RESULTS / task_id
    data = json.loads((base / "site" / "data.json").read_text())
    rdir = base / "renders"
    images = {}
    for key in data["records"]:
        g = rdir / f"{key}_grid.png"
        views = [rdir / f"{key}_{v}.png" for v in ("cond", "iso", "front", "side")]
        if all(p.exists() for p in views):
            from ppbench.v2.render import grid
            grid([str(p) for p in views], g, cols=2, cell=384)
            images[key] = {"grid": jpeg(g, 720, 78)}
        clay = rdir / f"{key}_clay_cond.png"
        if clay.exists():
            images.setdefault(key, {})["clay"] = jpeg(clay, 360, 78)
    cond = data["task"]["condition"]["image"]["path"]
    sheet = base / "pool_sheet" / "pool_sheet_all.png"
    payload = {"task": data["task"], "records": {k: slim(r) for k, r in data["records"].items()},
               "imagesim": data.get("imagesim", {}), "judges": data.get("judges", {}), "diversity": data.get("diversity", {}),
               "population": data.get("population"), "anchors": data.get("reference_anchors"),
               "images": images, "condition_image": jpeg(cond, 720, 82), "pool_sheet": jpeg(sheet, 1400, 80) if sheet.exists() else None}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    out = base / "site" / "office_chair_trials.html"
    out.write_text(html)
    print(out, f"{len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="swivel_office_chair")
    build(ap.parse_args().task)
