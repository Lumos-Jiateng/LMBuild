# Metrics v3.8: the audit of 2026-10-02

After the core set grew to 200 objects, every metric on which a system other than a closed-source API led, and every
score that looked out of line with the outputs, was audited. Four findings led to revisions. **Each revision is a
single rule applied identically to all systems; every system was re-scored with it; no score is adjusted for an
individual system.** Old record sets are kept next to the new ones, so every change can be compared design by design.

| metric | before | after (v3.8) | code | records |
|---|---|---|---|---|
| S.1–S.3 | v3.3, fixed 4 mm instrument | **L1 v3.4**, instrument scaled to each design | `ppbench/v2/spec_v34_l1.py` | `eval_v34l1/` |
| A.1–A.3 | v3.7, parts grounded by name | **L2 v3.8**, names confirmed by geometry | `ppbench/v2/afford/rules_v38.py`, `afford/grounding_v38.py` | `eval_v38l2/` |
| D.1–D.3 | v3.4.5 | v3.4.5 unchanged; **both judges on every design** | `judge_l3.py`, `spec_v34.py`, `spec_v344.py` | `eval_v345/` |
| R.1 | v3.5 | v3.5 unchanged | `spec_v35.py` | `eval_v35/` |
| R.2 | v3.5, unrecognised parts skipped | **R.2 v3.6**, role by name or geometry, every part counted | `ppbench/v2/spec_v36_l4r2.py` | `eval_v38p2/` |
| R.3 | v3.5, part presence by name | **R.3 v3.8**, presence uses verified roles | `ppbench/v2/spec_v38_p3.py` | `eval_v38p3/` |

`scripts/evaluate.sh` runs the whole chain; `scripts/export_scores.py` reads exactly these record sets.

## S.1–S.3: L1 v3.4, the instrument scales with the design

**Problem.** v3.3 used a fixed 4 mm voxel grid and absolute floors chosen for metre-scale furniture. A toy-scale design
(LEGO outputs are 6–17 cm tall) or a design built from thin parts is smaller than the instrument: a 1-stud brick is
two cells wide, its eroded core is empty, and a whole 2×2 brick (2.5 cm³) is below the 1 cm³ collision floor. Only
0–3 of 10 duplicate bricks injected into real BrickGPT / LegoACE outputs were detected.

**Rule (every design).** From the design itself: D = bounding-box diagonal, t = median over parts of the part's
smallest bounding-box extent.

    pitch = clip(min(D / 250, t / 4), 0.4 mm, 4 mm)      (coarsened by 1.25x, never beyond 4 mm, above 3e8 cells)
    s     = pitch / 4 mm

| constant | v3.3 | v3.4 |
|---|---|---|
| voxel pitch (also the contact tolerance) | 4 mm | 4 mm × s |
| collision floor | 1 cm³ | 1 cm³ × s³ (15.6 cells, as before) |
| interlock floor | 0.1 cm³ | 0.1 cm³ × s³ |
| fastening interface area | 1 cm² | 1 cm² × s² |
| joint-origin tolerance | 2 cm | 2 cm × s |
| ground-band floor | 5 mm | 5 mm × s (band = max(floor, 1 % of height), as before) |
| fraction rules (10 % / 25 % of the smaller part), tilt / push targets, 5× oversize rule | unchanged | unchanged |
| LDraw stud-stacking tolerance 1.5 mm (a brick-system constant) | unchanged | unchanged |

The pitch is never coarser than v3.3, so a design of metre-scale, thick parts is scored as before; a small design, or
one built from thin parts, is resolved with the same number of cells per object or per part thickness. Reference
anchors (stand and push targets, `anchors_v34l1.json`) are measured with the same rule.

**Validation.** Injected duplicates detected at true scale (v3.3 → v3.4): BrickGPT offroad_jeep 2/10 → 9/10,
dining_chair 0/10 → 10/10; LegoACE dining_chair 3/10 → 9/10, offroad_jeep 0/10 → 9/10; metre-scale agent designs
unchanged. BrickGPT stays collision-free (S.2 = 100) by construction. Newly detected collisions are real
interpenetrations of thin parts; newly detected disconnections are real gaps (e.g. 1.4–1.9 mm between a wheel and its
axle on a 13 cm model, the same ~1.5 % relative gap that v3.3 already flagged on metre-scale designs). Rank agreement
of the 30 systems, v3.3 vs v3.4: Spearman 0.94 (S.1), 0.97 (S.2), 0.98 (S.3).

## A.1–A.3: L2 v3.8, a part's name is a claim confirmed by geometry

**Problem.** v3.7's A.2 grounded a design whose parts carry names by name alone, while an unnamed design had to place
each part where the reference has that component. A system given the component list as its part schema
(Cube3D+CubePart) collected component credit for its labels (A.2 = 61.0, the highest of all systems) although its
named parts overlap the reference parts of the same name poorly (part-geometry F = 13.5 vs 36–46 for frontier APIs).

**Rule (every system).** A declared role is checked by the test already used for unnamed parts: at least half of the
part's surface within 5 % of the normalised diagonal of the reference's parts of that component, after the A.1 yaw
alignment.
- passes: verified, factor 1.0;
- fails but lies on another component: grounded there by geometry, factor 1.0;
- fails and lies on none: unverified, factor 0.5 (v3.7's existing value for "present but not doing its job");
- component without reference geometry: an internal component whose part is enclosed in the design is verified by
  placement (1.0), otherwise unverifiable (0.5);
- unnamed parts: geometry grounding as in v3.7.

A.2 instance credit = attachment (v3.7) × verification. A.1 and A.3 are unchanged from v3.7.

**Effect.** Cube3D+CubePart A.2 61.0 → 32.7. Frontier APIs lose 12–16 points (only 28–60 % of their named parts sit
where the reference has that component); open models lose 3–15; unnamed-mesh generators, BrickGPT and LegoACE are
unchanged.

## R.3: v3.8, presence uses the verified roles

R.3's `part_present` link becomes min(Σ verification factors, need) / need with the factors above. Chains, reach,
joints and the MuJoCo simulation are unchanged. Cube3D+CubePart 24.0 → 13.4; agents move by at most 1.2 points.

## R.2: v3.6, every part is counted

**Problem.** v3.5 dropped a part whose name gave no basis (no cited claim, Artiverse prior, reference part in place
or judged function) and skipped a design with no such part. Agents reach 98–100 % part coverage; PhysX-Anything, the
only generator that declares materials, was scored on 290 of 1,018 parts and 97 of its 188 outputs.

**Rule (every design that declares a library material).**
1. role by name, exactly as v3.5; the judged-function lookup also reads
   `ppbench/v2/lexicons/part_function_materials_v36_extra.json` (same prompt, model and settings as the v3.5 table,
   over part names v3.5 never saw);
2. if the name gives no basis, role by place: Level 2 geometric grounding (same normalisation and best yaw as A.1)
   gives the component, whose name is used as the role;
3. R.2 = mean of a_i over **all** parts; a part with no basis after 1–2 scores 0. Nothing is skipped.

Designs that declare no material: 0 by rule (tables show N/A). Agents move by at most 0.8 points; PhysX-Anything's
parts now have a basis in 95 % of cases and all of its outputs are scored (R.2 = 61.4).

## D.1–D.3: both judges on every design

D.1–D.3 average two VLM judges (Qwen2.5-VL-32B-Instruct, gemma-3-27b-it) and are calibrated on the mean of the two.
In the first scoring of the 150 new objects one judge had not run; it was re-run, and every rendered design is now
scored by both. The formula and the calibration (`docs/evaluation/l3_calibration_v345.json`) are unchanged.

## Tested and not adopted

**A name-aware D.2 prompt** that asks whether the design is an elegant instance of the named object (as the human
raters were asked). On the 7 held-out objects of the human study it agreed less with the human ratings than the
v3.4.5 prompt (MAE 0.260 vs 0.204; system-level Spearman 0.60 vs 0.83) and barely changed the text-to-3D generator
it targeted (Cube3D†(P) 82.2 → 79.2). Whether a design is the requested object is measured by D.3.
