# Evaluation

Every design is scored on twelve metrics in four groups (metrics **v3.8**; what changed after the audit of 2026-10-02
and why is in [`docs/evaluation/metrics_v38_changes.md`](evaluation/metrics_v38_changes.md)). Scores are in [0, 1] (tables show x100); a table cell is
the mean over tasks. An empty or unloadable design scores 0. A dimension that does not apply to a design (for
example D.2 when no judge answered) is left out of that mean and marked `skipped` in the record. A score computed
with a weaker instrument than intended is marked `degraded`, never `pass`.

| group | metric | record dim | spec | code | record dir |
|---|---|---|---|---|---|
| **S** Structure | S.1 connectivity | 1.1 | v3.4 | `ppbench/v2/spec_v34_l1.py` (on `spec_v33.py`) | `eval_v34l1/` |
| | S.2 collision | 1.2 | v3.4 | `spec_v34_l1.py` | same |
| | S.3 stability | 1.3 | v3.4 | `spec_v34_l1.py` | same |
| **A** Affordance | A.1 geometry | 2.1 | v3.7 (= v3.8) | `ppbench/v2/afford/rules.py` | `eval_v38l2/` |
| | A.2 parts | 2.2 | v3.8 | `afford/rules_v38.py`, `afford/grounding_v38.py` | same |
| | A.3 kinematics | 2.3 | v3.7 (= v3.8) | `afford/rules.py`, `afford/kin.py` | same |
| **D** Design | D.1 decomposition | 3.1 | v3.4.5 | `spec_v34.py` (raw halves), `spec_v344.py` + `calibrate_l3.py` (calibration) | `eval_v345/` |
| | D.2 aesthetics | 3.2 | v3.4.5 | `judge_l3.py`, `judge_views.py` | same |
| | D.3 structure alignment | 3.3 | v3.4.5 | `spec_v34.py`, `judge_l3.py` | same |
| **R** Realization | R.1 sequence | P.1 | v3.5 | `ppbench/v2/spec_v35.py` | `eval_v35/` |
| | R.2 material | P.2 | v3.6 | `spec_v36_l4r2.py`, `spec_v35.py`, `materials.py` | `eval_v38p2/` |
| | R.3 operability | P.3 | v3.8 | `spec_v38_p3.py`, `operability.py`, `sim_v35.py` (MuJoCo) | `eval_v38p3/` |

Nothing is written per object. The evaluator's task knowledge is the task's cited claims (`required_parts`,
`required_attributes`, `functional_subsystems`, `required_kinematics`), its reference object, and — for Level 2 — the
population of real instances of the category (precomputed as `results/v2/<task>/afford/sheet_v37.json`).

## Level 1 — Structure (spec v3.4, geometry only; declarations are never read)

Each design is voxelised per part (`ppbench/v2/voxel.py`) with an instrument scaled to the design (v3.4): with D the
bounding-box diagonal and t the median over parts of the smallest bounding-box extent,
`pitch = clip(min(D/250, t/4), 0.4 mm, 4 mm)` and s = pitch / 4 mm. Every length (contact tolerance, joint-origin
tolerance, ground band), area (fastening interface) and volume (collision and interlock floors) below is the
metre-scale value times s, s² or s³; a metre-scale design of thick parts gets s = 1, i.e. exactly v3.3. The values
quoted below are those at s = 1.

- **S.1 connectivity.** A connection needs physical evidence: interlocking material (≥ 0.1 cm³ core overlap), a
  declared joint that is fastened (its origin within 2 cm of both parts and ≥ 1 cm² shared interface), or a lattice
  stud for LDraw bricks. Proximity alone is not evidence. Scored on the number of breaks, with a whole-body term at
  one third.
- **S.2 collision.** Every interfacing pair counts, declared joints included, at a wider tolerance than contact.
- **S.3 stability.** Does it stand, and how hard must you push at its top edge to tip it over (critical tilt and
  push margin against per-task targets derived from the reference with the same scaled instrument,
  `results/v2/<task>/anchors_v34l1.json`).

## Level 2 — Affordance (spec v3.8, rule-based, no VLM)

- **A.1 geometry** = 0.6 · shape + 0.4 · part geometry. Shape is an F-score between point samples of the design and
  the reference after normalisation and the best of four yaw turns. Part geometry compares each claimed component's
  parts to the reference's labelled parts. Designs that declare no roles get the shape term only. Creating parts is
  reported, never scored, so Tier A, B and C are scored by the same rule.
- **A.2 parts** = 2/3 · visible components + 1/3 · internal components, over the claimed components (count-aware for
  plural claims). A component present but floating counts 1/2. A component that is there but not named is found by
  placement against the reference; an unnamed interior module is found by an air-flood from outside (1/2 credit).
  **v3.8:** a part's declared role is a claim, confirmed by the same placement test (at least half of its surface
  within 5 % of the normalised diagonal of the reference parts of that component, after the A.1 yaw). Confirmed:
  credit × 1; found on another component: regrounded there; on none: × 1/2. A component with no reference geometry
  counts as confirmed when its part is enclosed in the design (internal), otherwise × 1/2.
- **A.3 kinematics** = F1(target recall, joint precision). Targets are the reference's annotated joints (Artiverse,
  LEGO connector joints) plus the claimed motions of visible parts. A declared joint counts when it is on the right
  part with the right geometric relation (type, axis, sweep checked for collisions); a sound joint that no claim asks
  for counts 1/2.

## Level 3 — Design (spec v3.4.5, judge + computed, calibrated to a human study)

Two non-contestant VLM judges (Qwen2.5-VL-32B-Instruct and gemma-3-27b-it, temperature 0, mean of the two; every
design in the paper is scored by both) score
rendered sheets on 1–10 rubrics (`judge_l3.py`); they never see part lists, coordinates or which system made the
design. Sheets (`judge_views.py`): a four-view grey clay grid (3.2, 3.3) and a part-coloured plus exploded sheet
(3.1). The computed halves are in `spec_v34.py` (3.1: granularity, material consolidation, joint placement; 3.3:
part-correspondence IoU and symmetry match against the reference).

Each half is mapped onto the human scale with a clipped linear map fitted on 8 pre-declared **fit** objects of the
human study (`human_evaluation/`, 4 raters, 90 designs); 7 objects are held out for validation
(`docs/evaluation/l3_calibration_v345.json`, `..._heldout.json`, method in `details/l3_calibration_method.md`):

    J' = clip((judge − lo_J) / (hi_J − lo_J), 0, 1)        C' = clip((computed − lo_C) / (hi_C − lo_C), 0, 1)
    D.1 = (J' + C') / 2        D.2 = J'        D.3 = (3·J' + C') / 4

With one half missing the other calibrated half is used (`degraded` if only the computed half is left); D.2 is
`skipped` without a judge.

## Level 4 — Realization (spec v3.5)

- **R.1 sequence** = ½ E + ½ S. E is the fraction of parts placed, in the declared order, before the first state
  that cannot be executed (every placed body grounded and stable, connected parts treated as one solid, a straight
  approach for each part, never from below the floor). S averages over the declared steps the weakest body's critical
  tilt over the S.3 target (capped at 1; a failed step counts 0), divided by the number of parts. No sequence → 0.
- **R.2 material** = mean over parts of a_i: 1 if the part's declared material class is in its accepted set, 0.5 if
  in the same family, else 0. The accepted set comes from evidence first — the task's wiki material claims and
  quote-verified Wikipedia materials for the role, the role's ≥ 10 % classes across the Artiverse category
  (`lexicons/material_prior_v35.json`), and the reference parts in the same place — and otherwise from a functional
  judgement of the part name (`lexicons/part_function_materials_v35.json` + `part_function_materials_v36_extra.json`).
  **v3.6:** a part whose name gives no basis takes the role of the component it is grounded to by geometry (Level 2
  placement test); the mean runs over all parts, and a part with no basis after that scores 0 (nothing is skipped).
  No material declared → 0.
- **R.3 operability** = mean over capabilities of the weakest link. Capabilities come from the cited claims
  (`opsheets/<task>.json`): one per functional subsystem (parts present, power/motion paths ending on movable parts,
  control within reach and driving the mover), one per claimed motion (simulated in MuJoCo: rolls when pushed /
  moves through its range; kinematic check if the engine fails), human-use capabilities, and required parts no
  subsystem covers. No roles → 0. **v3.8:** "parts present" counts each part by its Level 2 verification factor
  (1 confirmed, 1/2 unconfirmed).

## Running it

`bash scripts/evaluate.sh [TASKS] [SYSTEM_SUBSTRING]` runs the whole chain on whatever is under
`results/v2/<task>/runs/` (and `external_core/` for generators), then writes `results/scores/my_scores.csv`. Set
`JUDGE_QWEN_URL` / `JUDGE_GEMMA_URL` to OpenAI-compatible endpoints of the two judges (`scripts/serve_judges.sh`) to get
D.1–D.3 as in the paper. Every builder caches by design stamp, so re-running scores only what is new.

Validation shipped with the release:
- `python scripts/export_scores.py` + `python scripts/make_tables.py --latex results/paper` regenerate
  `results/tables.md` and the paper's LaTeX tables (main 200-task table, stratified 50-task resamples and their
  correlations, metric audit) from the records; from the shipped `results/scores/all_scores.csv` they are
  byte-identical to `results/paper/*.tex`.
- `.venv_eval/bin/python -m ppbench.v2.calibrate_l3 heldout` reproduces `docs/evaluation/l3_calibration_v345_heldout.json`
  from the human ratings, the stored judge answers and the stored computed halves.
- `.venv_eval/bin/python -m ppbench.v2.golden` runs the hand-built golden scenes (one per check).

Detailed write-ups from the study are in `docs/evaluation/details/`: `metrics_spec.md` (the spec history),
`results_final_v345.md` (Levels 1 and 3), `results_level2_v37.md`, `results_level4_v35_tierB_draft.md`,
`results_allscope_v345.md` (every setting) and `l3_calibration_method.md`.
