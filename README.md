# LMBuild: Evaluating LLM Agents for Generating Buildable and Functional Structures

[**Project page**](https://lumos-jiateng.github.io/LMBuild/) · [**Paper**](https://lumos-jiateng.github.io/LMBuild/assets/LMBuild.pdf) · [**Dataset (Hugging Face)**](https://huggingface.co/datasets/Lumos-Jiateng/LMBuild)

LMBuild asks a system to **build a real, working object as an assembly of parts**. The object can be an office desk, a
wheelchair, a scissor car jack or a LEGO fire engine. The system places catalogue parts or creates new ones, and
declares what each part is, what it is made of, where the object moves and how it is put together. It works from a
name and an in-the-wild photo, through a tool environment, over up to three review rounds. The benchmark then scores
the result as a physical product on twelve metrics in four groups: **S**oundness, **A**ffordance, **D**esign and
**R**ealization.

The code calls the paper's interaction protocols *tiers*: `--tiers B` is the **Baseline Interaction Protocol**
(retrieve and create parts), `A` is **retrieval-only** and `C` is **creation-only**. The main setting is Tier B with
condition `name_only+image` (object name plus in-the-wild photo).

This repository holds everything needed to reproduce the paper's results and to run new systems:

| | what | where |
|---|---|---|
| 1 | **The 200 core tasks**: prompts (6 conditions), in-the-wild images, a mixed subpart pool, cited real-world knowledge (parts, attributes, subsystems, kinematics, each with Wikidata/Wikipedia evidence), the reference object with its joints, and the source bundles | `benchmark/` |
| 2 | **The environment and tools** (three tiers, shell and OpenAI-compatible transports, Claude Code / Codex / vLLM drivers) and **the evaluator** (all twelve metrics, the VLM judges, human-study calibration) | `ppbench/`, `scripts/`, `configs/` |
| 3 | **Results**: every score behind the paper's tables, the evaluator's per-task fixtures, the human study | `results/`, `human_evaluation/` |
| 4 | **Demos**: full agent trajectories (every tool call and observation), part-creation examples, final designs, renders, joint-motion, assembly and turntable GIFs | `demos/` (open `demos/index.html`) |
| 5 | **Designs**: all 6,858 scored designs of the 30 systems in the main setting, re-scorable | `designs/` |

Items 1, 4, 5, the evaluator fixtures in `results/v2/` and LMBuild-Full (2,549 objects) are hosted on
[Hugging Face](https://huggingface.co/datasets/Lumos-Jiateng/LMBuild); `scripts/download_data.py` puts them in place.

<p align="center">
  <img src="https://lumos-jiateng.github.io/LMBuild/assets/gifs/wheelchair__gpt-6-astra__B_turntable.gif" width="30%">
  <img src="https://lumos-jiateng.github.io/LMBuild/assets/media/joints/laptop__gpt-5.6-sol__lid_hinge.gif" width="30%">
  <img src="https://lumos-jiateng.github.io/LMBuild/assets/media/joints/kitchen_oven__gpt-6-astra__door_hinge.gif" width="30%">
</p>

## Quick start

```bash
# 1. environments: .venv_eval (Python 3.11) and .venv_render (Python 3.13, Blender's bpy), then a smoke test
PY311=python3.11 PY313=python3.13 bash scripts/setup.sh

# 2. data from Hugging Face: the 200-object benchmark and the evaluator fixtures (~12 GB);
#    add --designs (all scored designs), --demos, --full (LMBuild-Full) or --all
pip install huggingface_hub && python scripts/download_data.py

# 3. play an episode by hand: this is exactly what an agent sees and does
export PYTHONPATH=.
.venv_eval/bin/python -m ppbench.v2.session init --state /tmp/S.json --task office_desk --tier B \
    --condition name_only+image --rounds 3 --out /tmp/my_run --system me \
    --snapshot results/v2/office_desk/task_snapshot_core.json
.venv_eval/bin/python -m ppbench.v2.session prompt --state /tmp/S.json      # instructions, tools, task, image paths
.venv_eval/bin/python -m ppbench.v2.session call --state /tmp/S.json list_parts '{}'
```

`python scripts/smoke_test.py` loads all core tasks (pools and references) and plays a scripted episode with renders.

## Reproducing the paper

**Tables from the stored scores** (no GPU):

```bash
python scripts/make_tables.py --latex results/paper > results/tables.md   # from results/scores/all_scores.csv (38,412 scored designs)
.venv_eval/bin/python -m ppbench.v2.calibrate_l3 heldout # Level 3 calibration, validated on the held-out objects
```

**Running a system** (each writes `results/v2/<task>/runs/core_v2.4/<system>__<tier>__<condition>__r<rounds>__s0/`):

```bash
# Claude models through headless Claude Code (the paper's Claude arms); needs the `claude` CLI, logged in
.venv_eval/bin/python -m ppbench.v2.claude_batch tool --models claude-sonnet-5 --tiers B \
    --conditions name_only+image --rounds 3 --tasks office_desk,toilet --concurrency 2 --budget-usd 20

# GPT models through the Codex CLI (the paper's GPT arms)
.venv_eval/bin/python scripts/run_codex_episode.py --model gpt-6-astra --task office_desk --tier B

# open models served with vLLM (configs/open_models.tsv holds the paper's 13 models and their serving flags)
bash scripts/run_open_model.sh qwen3.5-27b 0 8101 --tasks office_desk --tiers B --conditions name_only+image
```

**Scoring** new episodes with all twelve metrics:

```bash
bash scripts/serve_judges.sh 0 1       # optional: the two Level 3 VLM judges (Qwen2.5-VL-32B, gemma-3-27b), vLLM
JUDGE_QWEN_URL=http://127.0.0.1:8111/v1 JUDGE_GEMMA_URL=http://127.0.0.1:8112/v1 bash scripts/evaluate.sh office_desk,toilet
python scripts/make_tables.py --csv results/scores/my_scores.csv
```

Without judge endpoints D.2 is skipped and D.1 / D.3 use their computed half (marked `degraded`); the other nine metrics
need only the CPU. [`docs/EVALUATION.md`](docs/EVALUATION.md) defines every metric.

This regenerates every table of the paper: the main 200-task table, original 50 vs new 150, the published 3-round
setting on the original 50, two disjoint stratified 50-object samples and 1,000 repeated draws (Pearson / Spearman
agreement of the 30 systems' scores), ranking validity, the metric-audit table, and the ablations. `results/paper/*.tex`
are its LaTeX output and are reproduced byte for byte. Scores use the final metrics **v3.8**
([`docs/evaluation/metrics_v38_changes.md`](docs/evaluation/metrics_v38_changes.md)).

**Settings in the paper**: main = all 200 core tasks, Tier B, `name_only+image`, round 1, seed 0 (for the original 50
tasks: the round-1 checkpoint of the 3-round trajectory; rows flagged `main200` in the score table). A missing design
scores 0. Ablations (original 50 tasks): the published setting (one 3-round trajectory, final design), Tier A / Tier C
(same trajectory), round checkpoints R1–R3, and prompt richness (`attributes+image`, `functional+image`, Tier B, one
round). Systems: 6 closed APIs (GPT-6 Astra, GPT-5.6 Sol, Claude Opus 5, Claude Fable
5.1, Claude Sonnet 5, Claude Haiku 4.5), 13 open models, and 8 domain-specific generators (11 rows with the Particulate
articulation variants).

## Layout

```
benchmark/                 the 200 core tasks                                    -> benchmark/README.md
  core_v2.json             all tasks (one JSON list; every path is relative to the repository root)
  tasks/<task>/            task.json, images/, reference_renders/, reference/, pool/, pool_sheet/, knowledge/
  assets/                  shared LDraw part meshes and Artiverse annotations for the references
ppbench/                   the package
  v2/env.py                the tool environment (tiers A/B/C, protocol v2.2); csg.py part creation; design.py
  v2/session.py, loop.py   shell transport (terminal agents) and OpenAI-compatible loop (served models)
  v2/claude_batch.py, core_batch.py, core_externals.py     drivers: Claude Code, vLLM models, generators
  v2/spec_v33.py ...       the evaluator (see docs/EVALUATION.md for which file computes which metric)
  v2/afford/               Level 2 (affordance) rules;  v2/judge_l3.py, judge_views.py  Level 3 judges
  baselines/               wrappers for the domain-specific generators (their upstream code is not shipped)
scripts/                   setup, smoke test, evaluation chain, drivers, score export, tables, packaging
configs/open_models.tsv    how each open model was served and driven
results/
  scores/all_scores.csv    one row per scored design: 12 metrics + status, every system and setting
  tables.md                the paper's tables, generated from it (scripts/make_tables.py)
  paper/                   the paper's LaTeX tables, generated from it (make_tables.py --latex results/paper)
  v2/<task>/               evaluator fixtures: frozen task snapshot, stability anchors, Level 2 population sheets,
                           and the judges' stored answers
  human_study/             computed records of the 90 human-study designs (calibration inputs)
human_evaluation/          the Level 3 human study: 90 designs, 4 raters, the rated sheets and the rating app
demos/                     trajectories, final designs, renders, GIFs                -> demos/README.md
docs/                      ENVIRONMENT.md, EVALUATION.md, benchmark format and construction, detailed result write-ups
```

## Data

Everything large lives in the Hugging Face dataset [`Lumos-Jiateng/LMBuild`](https://huggingface.co/datasets/Lumos-Jiateng/LMBuild):

| Hugging Face folder | local path | contents |
|---|---|---|
| `core/` | `benchmark/` | LMBuild-Core: 200 objects (task specs, images, references, pools, knowledge, shared assets) |
| `full/` | `benchmark/` | LMBuild-Full: 2,549 objects in the same format (`PPBENCH_CORE=benchmark/full_v2.json`) |
| `results/fixtures/` | `results/v2/` | evaluator fixtures for the 200 objects |
| `results/designs/` | `designs/` | all scored designs; `python scripts/stage_designs.py` links them into `results/v2/` for `evaluate.sh` |
| `results/demos/` | `demos/` | trajectories, final designs, renders, GIFs |
| `results/human_evaluation_renders/` | `human_evaluation/renders/` | the human-study sheets |

`python scripts/download_data.py [--full] [--designs] [--demos] [--all]` downloads and places them.
`benchmark/MANIFEST.json` gives a sha256 for every file. Sources and licences for each reference are listed in
[`DATA_LICENSES.md`](DATA_LICENSES.md) and per task in `benchmark/README.md`. Code is released under the MIT licence;
the data under CC BY-NC-SA 4.0 (required by the Fusion 360 Gallery source), with each object's upstream licence kept in
`reference.license`.

## Citation

```bibtex
@article{liu2026lmbuild,
  title   = {LMBuild: Evaluating LLM Agents for Generating Buildable and Functional Structures},
  author  = {Liu, Jiateng and Wang, Rushi and Qian, Cheng and Zhang, Xuejun and Li, Sun and Liu, Jiayu and
             Shen, Yifan and Cao, Xu and Yao, Jiarui and Li, Bingxuan and Sarikaya, Ruhi and Ji, Heng},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Using it from Claude Code

[`CLAUDE.md`](CLAUDE.md) tells Claude Code how to run episodes, score them and read the results. When Claude is the
**system under test**, run it through `ppbench.v2.claude_batch`. The batch driver starts each episode in its own working
directory outside the repository, with only the session command and file reading allowed, so the evaluated model never
sees this repository, its references or its scores.
