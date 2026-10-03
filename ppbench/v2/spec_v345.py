"""v3.4.5 Level 3 (P20): the v3.4.4 formula with the calibration fitted on the FIT objects of the human study only.

    .venv_eval/bin/python -m ppbench.v2.calibrate_l3 fit        # docs/evaluation/l3_calibration_v345.json
    .venv_eval/bin/python -m ppbench.v2.calibrate_l3 heldout    # validation on the TEST objects
    .venv_eval/bin/python -m ppbench.v2.spec_v345 build         # results/v2/*/eval_v345/
"""
from ppbench.v2.spec_v344 import build

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()
    ts = None if ns.tasks == "all" else [x.strip() for x in ns.tasks.split(",") if x.strip()]
    build(ts, ns.workers, ns.force, "v345")
