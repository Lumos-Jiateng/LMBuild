#!/usr/bin/env bash
# Score episodes under results/v2/<task>/runs/ with the paper's final evaluator (12 metrics, S/A/D/R).
#
#   bash scripts/evaluate.sh [TASKS] [SYSTEM_SUBSTRING]
#     TASKS             comma list of task ids, or "all" (default)
#     SYSTEM_SUBSTRING  only run keys containing this (default: every run key)
#
# Level 3's judge half needs two OpenAI-compatible VLM endpoints (the paper used Qwen2.5-VL-32B-Instruct and
# gemma-3-27b-it, served with vLLM; see scripts/serve_judges.sh). Point these at them:
#     JUDGE_QWEN_URL=http://127.0.0.1:8111/v1  JUDGE_GEMMA_URL=http://127.0.0.1:8112/v1
# Without them D.2 (judge-only) is skipped and D.1 / D.3 fall back to their computed half, marked "degraded".
#
# Order (each builder caches by design stamp, so re-running only scores what is new). Final metrics v3.8
# (docs/evaluation/metrics_v38_changes.md):
#   report collect-core --spec v3   geometry, voxels, raw metrics           -> eval_v3/
#   spec_v32, spec_v33              Level 1 base records                    -> eval_v32/, eval_v33/
#   spec_v34_l1                     S.1-S.3, design-scaled instrument (v3.4) -> eval_v34l1/
#   judge_views + judge_l3          judge sheets (Blender) and VLM answers  -> judge_views/, judge_l3/
#   spec_v34, spec_v345             D.1-D.3, calibrated halves (v3.4.5)     -> eval_v34/, eval_v345/
#   spec_v35                        R.1 (and v3.5 R.2/R.3, superseded)      -> eval_v35/
#   afford.rules score              Level 2 sheets and v3.7 base            -> eval_v37/
#   afford.rules_v38                A.1-A.3, names confirmed by geometry    -> eval_v38l2/
#   spec_v38_p3                     R.3, verified roles (v3.8)              -> eval_v38p3/
#   spec_v36_l4r2                   R.2, role by name or geometry (v3.6)    -> eval_v38p2/
#   scripts/export_scores.py        one CSV row per design                 -> results/scores/my_scores.csv
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS="${1:-all}"
FILTER="${2:-}"
PY="${PY:-.venv_eval/bin/python}"
WORKERS="${WORKERS:-8}"
export PYTHONPATH=. PPB_DECOMP_PROMPT=v1 PPB_JUDGE_SCOPE="${PPB_JUDGE_SCOPE:-all}" PPB_AFFORD_EXTRA="${PPB_AFFORD_EXTRA:-rounds,prompts}"
say() { echo "[$(date +%T)] $*"; }

ONLY=(); [ -n "$FILTER" ] && ONLY=(--only "$FILTER")
say "Level 1 raw metrics (collect-core)"
"$PY" -m ppbench.v2.report collect-core --spec v3 --rounds --tasks "$TASKS" --workers "$WORKERS" --filter '*' "${ONLY[@]}" | tail -n 3
"$PY" -m ppbench.v2.report collect-core --spec v3 --externals --tasks "$TASKS" --workers "$WORKERS" "${ONLY[@]}" | tail -n 1
for step in spec_v32 spec_v33; do
    say "$step"; "$PY" -m ppbench.v2.$step build --tasks "$TASKS" --workers "$WORKERS" | tail -n 1
done

# (task, key) list of every record to score with the v3.8 builders: all eval_v32 records of the selected tasks
KEYS="$(mktemp)"; trap 'rm -f "$KEYS"' EXIT
"$PY" - "$TASKS" "$FILTER" > "$KEYS" <<'PYKEYS'
import sys
from ppbench.v2.task import RESULTS, CORE
import json
tasks, flt = sys.argv[1], sys.argv[2]
ids = [t["task_id"] for t in json.loads(CORE.read_text())] if tasks == "all" else tasks.split(",")
for t in ids:
    for f in sorted((RESULTS / t / "eval_v32").glob("*.json")):
        if f.stem.startswith(("reference", "_")) or (flt and flt not in f.stem):
            continue
        print(f"{t}\t{f.stem}")
PYKEYS
say "Level 1 v3.4 (spec_v34_l1, design-scaled)"; "$PY" -m ppbench.v2.spec_v34_l1 build --scope "keys:$KEYS" --workers "$WORKERS" | tail -n 1

say "judge sheets"
"$PY" -m ppbench.v2.judge_views build --tasks "$TASKS" --workers "$WORKERS" | tail -n 1
if [ -n "${JUDGE_QWEN_URL:-}" ]; then
    say "judge qwen2.5-vl-32b"; "$PY" -m ppbench.v2.judge_l3 run --tasks "$TASKS" --model qwen2.5-vl-32b --base-url "$JUDGE_QWEN_URL" --workers 16 | tail -n 2
fi
if [ -n "${JUDGE_GEMMA_URL:-}" ]; then
    say "judge gemma-3-27b"; "$PY" -m ppbench.v2.judge_l3 run --tasks "$TASKS" --model gemma-3-27b --base-url "$JUDGE_GEMMA_URL" --workers 16 | tail -n 2
fi
[ -z "${JUDGE_QWEN_URL:-}${JUDGE_GEMMA_URL:-}" ] && say "no judge endpoints set: D.2 skipped, D.1/D.3 computed half only"

say "Level 3 (spec_v34 -> spec_v345)"
"$PY" -m ppbench.v2.spec_v34 build --tasks "$TASKS" --workers "$WORKERS" --force | tail -n 1
"$PY" -m ppbench.v2.spec_v345 build --tasks "$TASKS" --workers "$WORKERS" --force | tail -n 1
say "Level 4 (spec_v35)"
"$PY" -m ppbench.v2.spec_v35 build --tasks "$TASKS" --workers "$WORKERS" --scope all | tail -n 1
say "Level 2 (afford.rules sheets + v3.7 base)"
"$PY" -m ppbench.v2.afford.rules score --tasks "$TASKS" --workers "$WORKERS" --tiers A,B,C,A1,B1 | tail -n 1
say "Level 2 v3.8 (afford.rules_v38: names confirmed by geometry)"
"$PY" -m ppbench.v2.afford.rules_v38 score --keys-file "$KEYS" --workers "$WORKERS" | tail -n 1
say "R.3 v3.8 (spec_v38_p3: verified roles)"
"$PY" -m ppbench.v2.spec_v38_p3 build --keys-file "$KEYS" --workers "$WORKERS" | tail -n 1
say "R.2 v3.6 (spec_v36_l4r2: role by name or geometry, every part counted)"
"$PY" -m ppbench.v2.spec_v36_l4r2 build --keys-file "$KEYS" --workers "$WORKERS" | tail -n 1

say "score table"
"$PY" scripts/export_scores.py --results results/v2 --out results/scores/my_scores.csv
say "done: results/scores/my_scores.csv (python scripts/make_tables.py --csv results/scores/my_scores.csv)"
