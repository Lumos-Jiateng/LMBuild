"""Multi-file results site: index.html + per-design viewer data + large renders.

    .venv_eval/bin/python -m ppbench.v2.site_build2 --task swivel_office_chair [--workers 8]

Writes results/v2/<task>/site/v3/:
    index.html            the page (records inline)
    packs/<group>.js      what the page loads per system/tier/round group: viewer data, renders and evidence sheets
                          as data URIs; published assets are chosen under a size budget, seed 0 of every group first
    img/condition.jpg, img/pool.jpg
    designs/, img/<key>_<view>.jpg, evid/   local caches the packs are made from (not published)
Publish index.html with the files listed in files.json.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ppbench.v2.task import RESULTS

TEMPLATE = Path(__file__).with_name("site_template_v3.html")
SPEC = Path(__file__).resolve().parents[2] / "docs" / "evaluation" / "v2_evaluator_spec.json"   # per-metric evaluator specification
os.environ.setdefault("PPBENCH_VIEWER_FACES", "4000")    # every design ships 3D data; 4k faces keep ~260 designs under the size cap
TOTAL_CAP_MB = float(os.environ.get("PPBENCH_SITE_TOTAL_MB", "62.5"))   # the artifact limit is 64 MB per version; keep a margin
RENDER_SIDE, RENDER_Q = 512, 72        # renders re-encoded for the page (sources stay 768 px)
EVID_SIDE, EVID_Q = 900, 58            # evidence sheets re-encoded for the page


def _uri(path, side=None, quality=None):
    if side is None:
        return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    from io import BytesIO
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _pack(site, records, evidence, keys, budget_mb):
    """Group assets into packs/<group>.js under a size budget. Units, by priority: everything for seed 0 of each
    group; renders of other seeds; 3D data of other seeds; evidence sheets of other seeds (their verdicts are always
    in the index)."""
    from PIL import Image
    groups = {}
    for k in keys:
        it = records[k]["item"]
        gid = f"{it['system']}__{it['tier']}__r{it['rounds']}" if it["kind"] == "tool" else (it.get("system") or k)
        groups.setdefault(re.sub(r"[^A-Za-z0-9_.-]", "_", gid), []).append(k)
    units, thumbs = [], {}
    for gid, ks in sorted(groups.items()):
        ks.sort(key=lambda k: (records[k]["item"].get("seed") or 0, k))
        for i, k in enumerate(ks):
            first = i == 0
            dj = site / "designs" / f"{k}.js"
            if dj.exists():
                txt = dj.read_text()
                obj = txt[txt.index("]=") + 2:].rstrip().rstrip(";")
                units.append((0 if first else 2, gid, k, "designs", obj, len(obj)))
            imgs = {v: _uri(site / "img" / f"{k}_{v}.jpg", RENDER_SIDE, RENDER_Q) for v in ("cond", "iso", "front", "side")
                    if (site / "img" / f"{k}_{v}.jpg").exists()}
            if imgs:
                units.append((0 if first else 1, gid, k, "imgs", imgs, sum(map(len, imgs.values()))))
            if first and (site / "img" / f"{k}_cond.jpg").exists():
                im = Image.open(site / "img" / f"{k}_cond.jpg").convert("RGB")
                im.thumbnail((260, 260))
                tmp = site / "img" / f"{k}_thumb.jpg"
                im.save(tmp, "JPEG", quality=62, optimize=True)
                thumbs[k] = _uri(tmp)
            ev = {n: _uri(site / rel, EVID_SIDE, EVID_Q) for n, rel in ((evidence.get(k) or {}).get("web") or {}).items()
                  if n != "clay" and (site / rel).exists()}
            if ev:
                units.append((0 if first else 3, gid, k, "evid", ev, sum(map(len, ev.values()))))
    units.sort(key=lambda u: (u[0], u[1], u[2]))
    budget = budget_mb * 1e6 - sum(map(len, thumbs.values()))
    packs = {gid: {"designs": {}, "imgs": {}, "evid": {}} for gid in groups}
    used, dropped = 0, {}
    for pri, gid, k, kind, payload, size in units:
        if used + size > budget:
            dropped[kind] = dropped.get(kind, 0) + 1
            continue
        used += size
        packs[gid][kind][k] = payload
    out = site / "packs"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.js"):
        f.unlink()
    for gid, P in packs.items():
        designs = ",".join(f"{json.dumps(k)}:{obj}" for k, obj in P["designs"].items())
        (out / f"{gid}.js").write_text(f'(window.__PACKS=window.__PACKS||{{}})[{json.dumps(gid)}]={{"designs":{{{designs}}},'
                                       f'"imgs":{json.dumps(P["imgs"])},"evid":{json.dumps(P["evid"])}}};')
    assets = {k: {"pack": gid, "design": k in P["designs"], "img": k in P["imgs"], "evid": k in P["evid"]}
              for gid, P in packs.items() for k in groups[gid]}
    print(f"packs: {len(packs)} groups, {used / 1e6:.1f} MB of {budget_mb} MB; left out for size: {dropped or 'nothing'}")
    return assets, thumbs


def _one(args):
    task_id, key = args
    os.environ.setdefault("TMPDIR", "/tmp")
    from ppbench.v2 import evaluate as E  # noqa: F401  (import order: evaluate pulls analysis)
    from ppbench.v2.report import load_design
    from ppbench.v2.task import Task
    from ppbench.v2 import render_hq, viewer_export
    base = RESULTS / task_id
    site = base / "site" / "v3"
    rec = json.loads((base / "eval" / f"{key}.json").read_text())
    stamp_file = site / "img" / f"{key}.stamp"
    task = Task(task_id, snapshot=base / "task_snapshot.json")
    design = load_design(task, rec["item"])
    if rec["item"]["kind"] != "reference":
        from ppbench.v2.evaluate import prepare
        prepare(design, task.reference())      # same oracle scale / yaw the evaluator applied
    try:
        # The new cumulative Sol trajectory is the featured result. Preserve small
        # mechanical details in its interactive mesh; the static renders always use
        # the full source geometry.
        featured = rec["item"].get("system") in {"gpt-5.6-sol", "gpt-6-astra"}
        viewer_export.export(design, task_id, key, site / "designs", rec,
                             face_budget=16000 if featured else None,
                             min_faces=180 if featured else None)
        mode = None
        if not (stamp_file.exists() and stamp_file.read_text() == str(rec.get("stamp"))):
            _, mode = render_hq.render_design(design, task_id, site / "img" / key)
            stamp_file.write_text(str(rec.get("stamp")))
        return key, "ok", mode
    except Exception as e:  # one broken design must not stop the site
        return key, f"error {e!r}", None


def build(task_id, workers=8, skip_assets=False):
    from PIL import Image
    base = RESULTS / task_id
    site = base / "site" / "v3"
    (site / "img").mkdir(parents=True, exist_ok=True)
    data = json.loads((base / "site" / "data.json").read_text())
    keys = sorted(data["records"])
    if not skip_assets:
        with ProcessPoolExecutor(workers) as ex:
            for key, status, _ in ex.map(_one, [(task_id, k) for k in keys]):
                print(f"{key:60s} {status}", flush=True)
    cond = data["task"]["condition"]["image"]["path"]
    Image.open(cond).convert("RGB").resize((900, 900)).save(site / "img" / "condition.jpg", quality=88)
    sheet = base / "pool_sheet" / "pool_sheet_all.png"
    if sheet.exists():
        im = Image.open(sheet).convert("RGB")
        im.thumbnail((1760, 1760))
        im.save(site / "img" / "pool.jpg", quality=84)
    # evidence sheets the VLM judges saw, as web JPEGs: evid/<key>__<sheet>.jpg (cached by source mtime)
    (site / "evid").mkdir(exist_ok=True)
    evidence = {}
    for key, e in (data.get("evidence") or {}).items():
        imgs = {n: p for n, p in e["sheets"].items()}
        imgs.update({f"joint_{m['joint']}": m["image"] for m in e["motion"]})
        web = {}
        for n, src in imgs.items():
            dst = site / "evid" / f"{key}__{n}.jpg"
            if Path(src).exists():
                if not dst.exists() or dst.stat().st_mtime < Path(src).stat().st_mtime:
                    im = Image.open(src).convert("RGB")
                    im.thumbnail((1100, 1100))
                    im.save(dst, "JPEG", quality=70, optimize=True)
                web[n] = f"evid/{dst.name}"
        evidence[key] = {**e, "web": web}
    from ppbench.v2.site_build import slim
    # packs get whatever the index page (records, VLM verdicts, spec) and the two plain images leave under the cap
    probe = {"task": data["task"], "records": {k: slim(r) for k, r in data["records"].items()}, "vlm": data.get("vlm", {}),
             "evidence": {k: {"labels": e["labels"], "motion": e["motion"]} for k, e in evidence.items()},
             "imagesim": data.get("imagesim", {}), "spec": json.loads(SPEC.read_text()) if SPEC.exists() else []}
    index_mb = (len(json.dumps(probe, separators=(",", ":"))) + len(TEMPLATE.read_text()) + 400 * len(keys)) / 1e6
    plain_mb = sum((site / "img" / f).stat().st_size for f in ("condition.jpg", "pool.jpg") if (site / "img" / f).exists()) / 1e6
    budget = TOTAL_CAP_MB - index_mb - plain_mb - 0.3
    print(f"size cap {TOTAL_CAP_MB} MB: index ~{index_mb:.1f} MB, images {plain_mb:.1f} MB, packs get {budget:.1f} MB")
    assets, thumbs = _pack(site, data["records"], evidence, keys, budget)
    payload = {"task": data["task"], "records": {k: slim(r) for k, r in data["records"].items()},
               "imagesim": data.get("imagesim", {}), "vlm": data.get("vlm", {}),
               "evidence": {k: {"labels": e["labels"], "motion": [{kk: vv for kk, vv in m.items() if kk != "image"} for m in e["motion"]]}
                            for k, e in evidence.items()},
               "diversity": data.get("diversity", {}), "population": data.get("population"), "anchors": data.get("reference_anchors"),
               "assets": assets, "thumbs": thumbs,
               "spec": json.loads(SPEC.read_text()) if SPEC.exists() else []}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    (site / "index.html").write_text(html)
    files = sorted(str(p.relative_to(site)) for p in (site / "packs").glob("*.js")) + ["img/condition.jpg"] + (["img/pool.jpg"] if (site / "img" / "pool.jpg").exists() else [])
    (site / "files.json").write_text(json.dumps(files))
    total = sum((site / f).stat().st_size for f in files) + len(html)
    print(f"{site / 'index.html'}  {len(files)} files  {total / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="swivel_office_chair")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-assets", action="store_true")
    ns = ap.parse_args()
    build(ns.task, ns.workers, ns.skip_assets)
