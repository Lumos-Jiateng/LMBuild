"""Annotation server for the Level 3 human evaluation (stdlib only).

    python3 human_evaluation/server.py [--host 0.0.0.0] [--port 8790]

Each annotator opens  http://<host>:<port>/?t=<token>  (tokens in annotators.json, created on first run).
Progress for the organiser:  /admin?k=<admin token>;  unblinded CSV:  /export.csv?k=<admin token>.

Every save is appended to annotations/<annotator>.jsonl (the full history, never rewritten) and the
latest answer per item is kept in annotations/<annotator>.json (atomic replace).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import secrets
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
ANN = HERE / "annotations"
LOCK = threading.Lock()


def _load_set():
    return json.loads((STATIC / "set.json").read_text())


def _tokens():
    p = HERE / "annotators.json"
    if not p.exists():
        s = _load_set()
        p.write_text(json.dumps({"admin": secrets.token_urlsafe(9),
                                 "annotators": {a: secrets.token_urlsafe(9) for a in s["assignments"]}}, indent=1))
    return json.loads(p.read_text())


def _answers(ann):
    p = ANN / f"{ann}.json"
    return json.loads(p.read_text()) if p.exists() else {}


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _who(self, q):
        tok = (q.get("t") or [""])[0]
        for a, t in _tokens()["annotators"].items():
            if secrets.compare_digest(t, tok):
                return a
        return None

    def _admin(self, q):
        return secrets.compare_digest((q.get("k") or [""])[0], _tokens()["admin"])

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/session":
            a = self._who(q)
            if not a:
                return self._json({"error": "unknown link -- ask the organiser for your personal URL"}, 403)
            s = _load_set()
            items = s["assignments"][a]
            return self._json({"annotator": a, "items": items, "answers": _answers(a),
                               "tasks": s["tasks"], "designs": {i["design"]: s["designs"][i["design"]] for i in items},
                               "questions": s["questions"], "dim_order": s["dim_order"]})
        if u.path == "/api/progress":
            if not self._admin(q):
                return self._json({"error": "forbidden"}, 403)
            s = _load_set()
            out = {}
            for a, items in s["assignments"].items():
                ans = _answers(a)
                done = [i for i in items if i["item_id"] in ans]
                last = max((v.get("saved_at", "") for v in ans.values()), default="")
                out[a] = {"done": len(done), "total": len(items), "last_save": last,
                          "by_dim": {d: sum(1 for i in done if i["dim"] == d) for d in s["dim_order"]},
                          "per_dim": {d: sum(1 for i in items if i["dim"] == d) for d in s["dim_order"]}}
            return self._json(out)
        if u.path == "/export.csv":
            if not self._admin(q):
                return self._json({"error": "forbidden"}, 403)
            return self._export()
        if u.path == "/admin":
            self.path = "/admin.html"
        elif u.path in ("/", "/index.html"):
            self.path = "/index.html"
        if u.path.startswith("/set.json"):      # the full assignment list is not for browsing
            return self._json({"error": "forbidden"}, 403)
        return super().do_GET()

    def end_headers(self):
        if self.path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path != "/api/save":
            return self._json({"error": "not found"}, 404)
        a = self._who(q)
        if not a:
            return self._json({"error": "forbidden"}, 403)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except (ValueError, TypeError):
            return self._json({"error": "bad json"}, 400)
        s = _load_set()
        item = next((i for i in s["assignments"][a] if i["item_id"] == body.get("item_id")), None)
        if not item:
            return self._json({"error": "item not assigned to you"}, 400)
        score = body.get("score")
        if not isinstance(score, int) or not 1 <= score <= 5:
            return self._json({"error": "the score must be an integer 1-5"}, 400)
        rec = {"item_id": item["item_id"], "design": item["design"], "task": item["task"], "dim": item["dim"],
               "score": score, "seconds": body.get("seconds"), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with LOCK:
            ANN.mkdir(exist_ok=True)
            with open(ANN / f"{a}.jsonl", "a") as f:
                f.write(json.dumps({"annotator": a, **rec}) + "\n")
            cur = _answers(a)
            cur[item["item_id"]] = rec
            tmp = ANN / f"{a}.json.tmp"
            tmp.write_text(json.dumps(cur, indent=1))
            tmp.replace(ANN / f"{a}.json")
        return self._json({"ok": True, "done": len(cur)})

    def _export(self):
        s = _load_set()
        key = json.loads((HERE / "key.json").read_text())
        dz = {d["id"]: d for d in key["designs"]}
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["annotator", "item_id", "design", "task", "system", "role", "eval_key", "dim", "dim_name",
                    "score_1_5", "score_norm", "seconds", "saved_at"])
        for a in s["assignments"]:
            for iid, r in sorted(_answers(a).items()):
                d = dz[r["design"]]
                w.writerow([a, iid, r["design"], r["task"], d["system"], d["role"], d["key"], r["dim"],
                            s["questions"][r["dim"]]["name"], r["score"], round((r["score"] - 1) / 4, 4),
                            r.get("seconds"), r["saved_at"]])
        b = buf.getvalue().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", "attachment; filename=human_eval_l3.csv")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8790)
    ns = ap.parse_args()
    tok = _tokens()
    print(f"serving on http://{ns.host}:{ns.port}", flush=True)
    for a, t in tok["annotators"].items():
        print(f"  {a}: /?t={t}", flush=True)
    print(f"  admin: /admin?k={tok['admin']}", flush=True)
    ThreadingHTTPServer((ns.host, ns.port), H).serve_forever()
