# Working in this repository (for Claude Code)

This is the LMBuild release: a benchmark where an agent builds an articulated object from parts through a tool
environment, and an evaluator scores the result on twelve metrics. Read `README.md` first. `docs/ENVIRONMENT.md` and
`docs/EVALUATION.md` describe the environment and the metrics.

## Setup and checks

- Python: `.venv_eval/bin/python` (3.11), with `PYTHONPATH=.`. Rendering runs through `.venv_render/bin/python`
  (Python 3.13 with `bpy`); override that path with `PPBENCH_BPY`. Create both with `bash scripts/setup.sh`.
- Health check: `PYTHONPATH=. .venv_eval/bin/python scripts/smoke_test.py` loads all core tasks and plays one scripted
  episode. Add `--no-render` if Blender is unavailable.
- Golden scenes: `PYTHONPATH=. .venv_eval/bin/python -m ppbench.v2.golden`. There is one hand-built scene per check, and
  every line must read `ok`.

## Running episodes

- **You as the system under test.** Never play an evaluated episode from inside this repository. Your context would
  include this file, the references and the scores. Use the isolated driver instead:
  `.venv_eval/bin/python -m ppbench.v2.claude_batch tool --models <claude model id> --tasks <t1,t2> --tiers B
  --conditions name_only+image --rounds 3 --concurrency 1 --budget-usd <cap>`.
  - Each episode runs in its own directory under `~/.cache/ppbench_claude_iso/`, with auto-memory off. The only allowed
    tools are the session command and `Read`.
  - Cost and session ids go to `claude_run.json`. An interrupted batch resumes; it never reruns a paid episode.
- **Interactive or debug episodes.** Use `ppbench.v2.session` (`init`, `prompt`, `call`, `finish`) with `--out` outside
  `results/`, so the output never mixes with scored runs.
- Other systems:
  - Codex: `scripts/run_codex_episode.py`.
  - Open models: `scripts/run_open_model.sh`, using the settings in `configs/open_models.tsv`.
  - Domain generators: `ppbench/v2/core_externals.py`. They need their upstream code under `third_party/`.
- Episodes live in `results/v2/<task>/runs/core_v2.4/<system>__<tier>__<condition>__r<rounds>__s<seed>/`.
  - `trace.json` records every call.
  - `rounds/round_<n>/` holds one immutable checkpoint per submit.
  - Treat finished episodes as read-only. Move a bad one aside; never delete it.

## Scoring

- `bash scripts/evaluate.sh <tasks|all> [system substring]` runs the whole chain and writes
  `results/scores/my_scores.csv`. `python scripts/make_tables.py --csv <csv>` turns that into tables.
- Level 3 needs the two judge endpoints: `JUDGE_QWEN_URL` and `JUDGE_GEMMA_URL`, served with `scripts/serve_judges.sh`.
  - Without them, D.2 is `skipped` and D.1/D.3 are `degraded`. Say so when reporting numbers.
  - Keep `PPB_DECOMP_PROMPT=v1`, the adopted 3.1 prompt. That is the default here.
- The paper's numbers are in `results/scores/all_scores.csv` and `results/tables.md`.
  - Main setting (paper Table 2): all 200 tasks, Tier B, `name_only+image`, round 1 — the rows with `main200` set
    (rank 1 preferred; `make_tables.py` applies the rule). Metrics v3.8 (`docs/evaluation/metrics_v38_changes.md`).
  - Published 50-task setting: Tier B, `name_only+image`, `trajectory_rounds=3`, `checkpoint=final`.
  - Generators use `tier=ext`.
- The evaluator reads its per-task fixtures from `results/v2/<task>/`:
  - `task_snapshot_core.json` holds the task exactly as the agents saw it.
  - `anchors_v34l1.json` holds the stability targets (Level 1 v3.4; `anchors_v33.json` is the v3.3 version).
  - `afford/sheet_v37.json` holds the Level 2 population sheets.
  - `judge_l3/*.json` holds the stored judge answers.
  - Do not regenerate these fixtures: rebuilding the population sheets needs the extended set, which is not shipped.

## Conventions

- World frame: metres, +Z up, ground at z = 0, front toward −Y.
- Designs: `design.json` + `design.glb` (`ppbench/v2/design.py`).
- Nothing in the evaluator is written per object. Task knowledge comes only from the task's cited claims, its reference,
  and the category population. Keep new metrics generic too.
- A result computed with a weaker instrument is `degraded`, never `pass`. A skipped dimension is left out of a mean, not
  counted as 0.
- Machine-heavy steps are parallel (`--workers`). Rendering is CPU Cycles by default and uses `RENDER_THREADS`
  threads per Blender process.
