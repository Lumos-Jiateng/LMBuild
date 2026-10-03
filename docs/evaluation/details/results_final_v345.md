# LMBuild final results, spec v3.4.5 (Level 3 human-calibrated, validated on held-out objects)

These are the main-setting results to cite from 2026-09-23. They supersede `results_final_v344.md` and
`results_final_v342.md`: Level 1 is unchanged, and Level 3 is replaced.
- Every setting (rounds, Tier C, prompt ablations, paired shifts): `results_allscope_v345.md`.
- Records: `results/v2/*/eval_v345/`.
- Method and validation, stand-alone: `l3_calibration_method.md`.
- Policy: P20 in `metrics_feedback.md`.

## Metrics

| dim | definition |
|---|---|
| 1.1 connectivity | mean(anchored volume fraction, break score, fastening-spec rate) (v3.3) |
| 1.2 collision | 1 − colliding / interfacing pairs (v3.3) |
| 1.3 stability | mean(stand score, push score) against the reference (v3.3) |
| 3.1 decomposition | (J′ + C′)/2: calibrated VLM judge + calibrated rule-based signal |
| 3.2 aesthetics | J′: calibrated VLM judge on clay views, with no reference |
| 3.3 structure alignment | (3·J′ + C′)/4: calibrated VLM judge + calibrated rule-based signal |

Scores are ×100 and the mean over tasks. An empty or unloadable design scores 0. N = tasks.

## Method: human-calibrated automatic evaluation of Level 3

**Signals.** Every design x gets two raw signals per dimension. Both were defined before any human rating was
collected.

- **Judge.** Two VLM judges that are not systems under test: Qwen2.5-VL-32B-Instruct and gemma-3-27b-it. Each
  scores a fixed 1–10 rubric with written anchors, at temperature 0:
  - 3.1 decomposition, on the part-coloured and exploded sheet plus the photo: boundaries, granularity, separability;
  - 3.2 aesthetics, on four grey-clay views with no reference: proportion, coherence, refinement;
  - 3.3 structure alignment, on the photo plus the clay views: inventory, placement, relative scale, detail fidelity.

  `judge(x) = mean over judges and criteria of (s − 1)/9`.
- **Computed (rule-based) signal.**
  - `comp_3.1 = mean(granularity, consolidation, joint placement)`, where
    - granularity = 2^−|log2(n_parts/n_ref_parts)|,
    - consolidation = 1 − same-material rigid pairs/(n_parts − 1),
    - joint placement = 1 − rejected/declared joints.
  - `comp_3.3 = mean(part IoU vs the reference, symmetry match)`.
  - 3.2 has no computed signal.

**Human study.** 4 annotators rated 90 designs per dimension on a 1–5 scale: 6 systems × 15 objects. The systems are
two closed APIs, two open models and two domain generators. There are 30 double-rated items per dimension. The
target is `H = mean over raters of (r − 1)/4`. Inter-annotator Spearman on the double-rated items is 3.1 0.69 / 3.2 0.52 / 3.3 0.73.

**Object split.** The 15 rated objects form 4 categories. The split was declared before any score was examined:
alphabetical order within each category, 1st and 3rd objects to **fit**, 2nd and 4th to **test**.
- fit (8): bed_frame, bench_drill_press, desk_lamp, dining_table, offroad_jeep, refrigerator, scissor_car_jack, wheelchair
- test (7): chest_of_drawers, oscillating_fan, oscillating_steam_engine, passenger_car, stepladder, swivel_office_chair, wheeled_excavator

All fitting uses the fit objects only (48 items per dimension). The test objects are used only for validation.

**Calibration (2 parameters per signal).** Each signal h is mapped linearly onto the human scale and clipped:

    h'(x) = clip( (h(x) − lo) / (hi − lo), 0, 1 )
    b = sd(H) / sd(h),  a = mean(H) − b·mean(h),  lo = −a/b,  hi = (1 − a)/b      (on the fit items)

On the fit items, h' has the human mean and standard deviation. We chose this moment matching over least squares,
because least squares shrinks predictions toward the mean (its predictions have about half the human spread).

**Weighting the rule-based signal (fitted).** The score is an integer-ratio blend of the calibrated signals:

    S(x) = (a·J'(x) + b·C'(x)) / (a + b)

(a:b) is chosen from {1:0, 4:1, 3:1, 2:1, 3:2, 1:1, 2:3, 1:2, 1:3, 0:1}. The choice minimises leave-one-object-out
MAE within the fit objects. The result is **3.1 = 1:1** and **3.3 = 3:1**. 3.2 is judge-only. A continuous
least-squares weighting on the fit objects gives about 0.58:0.42 for 3.1 and 0.79:0.21 for 3.3, consistent with
these ratios. If one signal is missing, the score is the other calibrated signal. An empty or unloadable design
scores 0.

| dim | judge lo → hi | computed lo → hi | judge : computed (chosen on fit objects) | fit items |
|---|---|---|---|---|
| 3.1 | 0.361 → 0.764 | 0.128 → 1.025 | 1:1 | 48 |
| 3.2 | 0.374 → 0.725 | – | 1:0 | 48 |
| 3.3 | 0.450 → 0.777 | 0.154 → 0.606 | 3:1 | 48 |

Leave-one-object-out MAE of each candidate ratio on the fit objects:

| ratio (judge:computed) | 1:0 | 4:1 | 3:1 | 2:1 | 3:2 | 1:1 | 2:3 | 1:2 | 1:3 | 0:1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3.1 LOO MAE on fit objects | 0.246 | 0.225 | 0.220 | 0.213 | 0.210 | **0.209** | 0.210 | 0.214 | 0.218 | 0.234 |
| 3.3 LOO MAE on fit objects | 0.213 | 0.206 | **0.205** | 0.205 | 0.209 | 0.217 | 0.226 | 0.232 | 0.241 | 0.272 |

## Validation on the held-out objects

**Pre-declared split: fit 8 objects → test 7 objects** (42 designs per dimension). The comparison is the v3.4.3
formula on the same raw signals: uncalibrated, with judge weights 0.3 / 1.0 / 0.7.

| dim | model | MAE ↓ | item Spearman ↑ | pairwise (same object) ↑ | system MAE ↓ | system Spearman ↑ |
|---|---|---|---|---|---|---|
| 3.1 | v3.4.3 (uncalibrated 0.3/1.0/0.7 judge) | 0.230 | 0.45 | 0.74 | 0.136 | 0.77 |
| 3.1 | **v3.4.5 (calibrated)** | 0.204 | 0.52 | 0.75 | 0.098 | 0.77 |
| 3.2 | v3.4.3 (uncalibrated 0.3/1.0/0.7 judge) | 0.225 | 0.60 | 0.63 | 0.158 | 0.83 |
| 3.2 | **v3.4.5 (calibrated)** | 0.204 | 0.60 | 0.63 | 0.148 | 0.83 |
| 3.3 | v3.4.3 (uncalibrated 0.3/1.0/0.7 judge) | 0.254 | 0.74 | 0.82 | 0.178 | 0.94 |
| 3.3 | **v3.4.5 (calibrated)** | 0.239 | 0.64 | 0.82 | 0.146 | 0.94 |

System means on the test objects (×100):

| system | 3.1 human / v3.4.3 / v3.4.5 | 3.2 human / v3.4.3 / v3.4.5 | 3.3 human / v3.4.3 / v3.4.5 |
|---|---|---|---|
| GPT-6 Astra | 84 / 84 / 80 | 86 / 63 / 72 | 79 / 69 / 79 |
| Claude Sonnet 5 | 93 / 86 / 85 | 77 / 65 / 80 | 80 / 66 / 74 |
| Qwen3.5-27B | 50 / 81 / 77 | 45 / 61 / 66 | 46 / 62 / 66 |
| Gemma-4-31B-IT | 61 / 67 / 60 | 32 / 56 / 55 | 38 / 61 / 62 |
| PartPacker | 70 / 57 / 59 | 61 / 65 / 78 | 61 / 63 / 71 |
| BrickGPT | 4 / 28 / 13 | 23 / 40 / 12 | 9 / 50 / 36 |

**Robustness: 50 random stratified object splits.** Each split keeps half of every category for fitting. The same
protocol runs inside every split, including the ratio choice. Numbers are the mean ± sd on the held-out objects.

| dim | model | MAE ↓ | item Spearman ↑ | system MAE ↓ | system Spearman ↑ | ratios picked (of 50) |
|---|---|---|---|---|---|---|
| 3.1 | v3.4.3 | 0.220 ± 0.016 | 0.484 ± 0.063 | 0.108 ± 0.026 | 0.827 ± 0.094 |  |
| 3.1 | **calibrated** | 0.214 ± 0.018 | 0.513 ± 0.065 | 0.100 ± 0.026 | 0.814 ± 0.090 | 1:1×14, 2:3×12, 3:2×7, 3:1×5, 1:2×4, 2:1×4, 0:1×2, 1:3×2 |
| 3.2 | v3.4.3 | 0.235 ± 0.011 | 0.659 ± 0.048 | 0.157 ± 0.007 | 0.827 ± 0.063 |  |
| 3.2 | **calibrated** | 0.209 ± 0.016 | 0.658 ± 0.048 | 0.107 ± 0.027 | 0.825 ± 0.061 | 1:0×50 |
| 3.3 | v3.4.3 | 0.264 ± 0.014 | 0.656 ± 0.046 | 0.175 ± 0.021 | 0.943 ± 0.069 |  |
| 3.3 | **calibrated** | 0.235 ± 0.019 | 0.674 ± 0.065 | 0.139 ± 0.042 | 0.930 ± 0.070 | 1:0×17, 2:1×15, 4:1×8, 3:1×6, 3:2×2, 1:1×1, 1:3×1 |

**Reading.**
- **Absolute agreement improves on unseen objects.** Held-out MAE drops in every dimension, and the system-level
  error drops by a quarter to a third for 3.2 and 3.3.
- **Ranking is mostly preserved.** Calibration is monotone within each signal. On the pre-declared split, 3.3's item
  Spearman falls from 0.74 to 0.64, while over 50 random splits it rises from 0.66 to 0.67.
- **The ratio is weakly identified.** In random splits, 3.1 picks 1:1 or 2:3 most often and 3.3 picks 1:0 to 4:1.
  The held-out error curve is flat around the chosen ratios.
- **Residual ranking errors come from the judges, not the scale.** On the test objects:
  - 3.2 rates Gemma-4 and Qwen3.5 well above humans (humans 32 and 45);
  - 3.1 over-rates Qwen3.5-27B through the lenient computed signal.

## 1. Baseline — Tier B, name_only+image, 3 rounds (final design)

Domain generators have no tier and no rounds: one output per task. Brick generators (BrickGPT, LegoACE, BrickNet) can only use existing bricks, the closest thing to Tier A; the mesh generators create free geometry, the closest thing to Tier C.

| system | 1.1 | 1.2 | 1.3 | 3.1 | 3.2 | 3.3 | built % | judged % | N |
|---|---|---|---|---|---|---|---|---|---|
| **Closed-source APIs** | | | | | | | | | |
| GPT-6 Astra | 92.6 | 100.0 | 92.9 | 76.2 | 79.4 | 82.8 | 100 | 100 | 50 |
| GPT-5.6 Sol | 91.2 | 100.0 | 90.3 | 78.5 | 77.5 | 74.3 | 100 | 100 | 50 |
| Claude Opus 5 | 95.2 | 100.0 | 91.5 | 75.7 | 76.6 | 77.5 | 100 | 100 | 50 |
| Claude Fable 5.1 | 90.4 | 100.0 | 89.6 | 74.7 | 78.4 | 77.7 | 100 | 100 | 50 |
| Claude Sonnet 5 | 85.4 | 100.0 | 85.3 | 77.2 | 69.4 | 74.0 | 100 | 100 | 50 |
| Claude Haiku 4.5 | 90.3 | 98.3 | 73.3 | 73.3 | 58.4 | 63.4 | 100 | 100 | 50 |
| **Open-source models** | | | | | | | | | |
| Qwen3.5-27B | 89.4 | 98.3 | 81.8 | 70.8 | 53.1 | 64.1 | 100 | 100 | 50 |
| Qwen3.5-35B-A3B | 79.3 | 94.9 | 77.8 | 70.7 | 49.1 | 60.0 | 96 | 100 | 50 |
| gpt-oss-120B | 81.1 | 99.7 | 70.9 | 66.1 | 49.7 | 54.2 | 100 | 100 | 50 |
| Gemma-4-31B-IT | 78.0 | 92.8 | 75.8 | 64.8 | 52.5 | 61.2 | 98 | 100 | 50 |
| Qwen3-VL-32B | 58.1 | 67.0 | 51.1 | 46.9 | 33.5 | 41.5 | 76 | 100 | 50 |
| Qwen3-VL-30B-A3B | 72.2 | 70.2 | 46.1 | 52.8 | 37.8 | 47.0 | 96 | 100 | 50 |
| Qwen3-VL-8B | 63.2 | 54.7 | 54.8 | 33.4 | 29.1 | 32.7 | 78 | 100 | 50 |
| Qwen3-4B-Instruct | 85.8 | 59.8 | 65.9 | 45.1 | 23.5 | 36.1 | 100 | 100 | 50 |
| InternVL3.5-38B | 60.4 | 87.3 | 59.8 | 56.9 | 40.3 | 48.6 | 96 | 100 | 50 |
| InternVL3.5-8B | 48.5 | 53.3 | 46.2 | 34.1 | 30.7 | 32.9 | 74 | 100 | 50 |
| MiniCPM-V-4.5 | 56.0 | 75.4 | 41.8 | 46.2 | 31.0 | 33.3 | 92 | 100 | 50 |
| ERNIE-4.5-VL-28B-A3B | 47.5 | 59.8 | 34.9 | 41.2 | 32.0 | 37.7 | 74 | 100 | 50 |
| Ministral-3-8B | 72.1 | 92.6 | 56.7 | 60.9 | 39.6 | 51.5 | 100 | 100 | 50 |
| **Domain generators, image input** | | | | | | | | | |
| PartCrafter (K=15) | 89.0 | 89.9 | 25.3 | 62.4 | 50.7 | 64.4 | 100 | 100 | 50 |
| ↳ + Particulate | 82.5 | 100.0 | 19.3 | 30.6 | 33.8 | 54.5 | 100 | 100 | 50 |
| PartCrafter (K=8) | 73.6 | 93.2 | 29.0 | 65.0 | 49.6 | 63.6 | 100 | 100 | 50 |
| ↳ + Particulate | 79.9 | 100.0 | 27.9 | 39.2 | 42.7 | 59.7 | 100 | 100 | 50 |
| PartPacker | 70.3 | 97.8 | 65.6 | 56.7 | 66.1 | 72.0 | 100 | 100 | 50 |
| ↳ + Particulate | 84.4 | 100.0 | 65.3 | 42.2 | 60.4 | 66.8 | 100 | 100 | 50 |
| PhysX-Anything | 58.7 | 97.8 | 50.9 | 49.6 | 51.8 | 55.9 | 100 | 100 | 48 |
| **Domain generators, text input (name_only text)** | | | | | | | | | |
| Cube3D → CubePart | 88.6 | 87.1 | 52.2 | 57.0 | 68.6 | 63.8 | 100 | 100 | 50 |
| ↳ + Particulate | 78.0 | 100.0 | 80.4 | 43.3 | 85.7 | 64.8 | 100 | 100 | 50 |
| BrickGPT | 99.1 | 100.0 | 83.9 | 18.0 | 16.6 | 33.6 | 100 | 100 | 50 |
| LegoACE | 94.0 | 100.0 | 74.8 | 15.3 | 11.8 | 34.3 | 100 | 100 | 50 |

## 3. Ablation 1a — Tier A (catalog only), name_only+image, 3 rounds

| system | 1.1 | 1.2 | 1.3 | 3.1 | 3.2 | 3.3 | built % | judged % | N |
|---|---|---|---|---|---|---|---|---|---|
| **Closed-source APIs** | | | | | | | | | |
| Claude Opus 5 | 88.2 | 100.0 | 85.9 | 78.6 | 62.0 | 73.8 | 100 | 100 | 20 |
| Claude Fable 5.1 | 90.2 | 100.0 | 91.0 | 75.5 | 69.9 | 78.8 | 100 | 100 | 20 |
| Claude Sonnet 5 | 84.8 | 100.0 | 85.7 | 74.7 | 69.2 | 74.6 | 100 | 100 | 50 |
| Claude Haiku 4.5 | 86.9 | 98.4 | 76.7 | 76.1 | 51.7 | 64.1 | 100 | 100 | 50 |
| **Open-source models** | | | | | | | | | |
| Qwen3.5-27B | 82.2 | 99.6 | 88.2 | 74.5 | 63.0 | 70.0 | 100 | 100 | 50 |
| Qwen3.5-35B-A3B | 79.6 | 95.4 | 74.3 | 70.7 | 50.4 | 63.6 | 96 | 100 | 50 |
| gpt-oss-120B | 76.1 | 99.9 | 72.0 | 70.9 | 58.2 | 61.4 | 100 | 100 | 50 |
| Gemma-4-31B-IT | 72.0 | 98.1 | 84.1 | 70.9 | 60.6 | 70.8 | 100 | 100 | 50 |
| Qwen3-VL-32B | 63.3 | 90.2 | 52.8 | 62.8 | 45.4 | 55.8 | 92 | 100 | 50 |
| Qwen3-VL-30B-A3B | 72.0 | 92.5 | 68.2 | 64.2 | 51.2 | 62.3 | 100 | 100 | 50 |
| Qwen3-VL-8B | 64.7 | 78.4 | 55.0 | 51.8 | 47.3 | 51.3 | 98 | 100 | 50 |
| Qwen3-4B-Instruct | 67.5 | 78.2 | 42.6 | 44.1 | 43.4 | 43.7 | 86 | 100 | 50 |
| InternVL3.5-38B | 57.3 | 96.2 | 65.6 | 64.9 | 50.1 | 56.7 | 100 | 100 | 50 |
| InternVL3.5-8B | 57.1 | 88.3 | 55.9 | 55.4 | 43.8 | 43.2 | 94 | 100 | 50 |
| MiniCPM-V-4.5 | 55.0 | 77.8 | 49.0 | 53.1 | 36.6 | 46.4 | 92 | 100 | 50 |
| ERNIE-4.5-VL-28B-A3B | 49.5 | 67.0 | 36.5 | 42.6 | 41.3 | 39.2 | 80 | 100 | 50 |
| Ministral-3-8B | 71.2 | 97.3 | 50.6 | 65.0 | 42.0 | 52.9 | 100 | 100 | 50 |

## 4. Ablation 1b — Tier C (create only, no catalog), name_only+image, 3 rounds

| system | 1.1 | 1.2 | 1.3 | 3.1 | 3.2 | 3.3 | built % | judged % | N |
|---|---|---|---|---|---|---|---|---|---|
| **Closed-source APIs** | | | | | | | | | |
| Claude Opus 5 | 85.7 | 100.0 | 94.2 | 78.4 | 66.3 | 71.5 | 100 | 100 | 20 |
| Claude Fable 5.1 | 83.3 | 100.0 | 93.9 | 78.5 | 79.4 | 72.6 | 100 | 100 | 20 |
| Claude Sonnet 5 | 87.8 | 99.6 | 88.3 | 72.6 | 62.0 | 68.9 | 100 | 100 | 50 |
| Claude Haiku 4.5 | 92.1 | 87.5 | 90.0 | 64.6 | 41.8 | 50.9 | 100 | 100 | 50 |
| **Open-source models** | | | | | | | | | |
| Qwen3.5-27B | 88.1 | 95.4 | 85.2 | 65.2 | 37.8 | 58.0 | 100 | 100 | 50 |
| Qwen3.5-35B-A3B | 82.0 | 84.0 | 72.5 | 61.3 | 27.1 | 50.1 | 98 | 100 | 50 |
| gpt-oss-120B | 87.4 | 95.7 | 87.6 | 60.0 | 36.7 | 47.3 | 100 | 100 | 50 |
| Gemma-4-31B-IT | 82.5 | 87.1 | 79.5 | 61.0 | 44.2 | 55.5 | 100 | 100 | 50 |
| Qwen3-VL-32B | 49.3 | 42.9 | 41.2 | 29.1 | 17.4 | 24.9 | 60 | 100 | 50 |
| Qwen3-VL-30B-A3B | 68.5 | 46.5 | 31.9 | 41.2 | 18.2 | 33.3 | 82 | 100 | 50 |
| Qwen3-VL-8B | 63.8 | 42.4 | 52.4 | 26.7 | 14.6 | 24.6 | 72 | 100 | 50 |
| Qwen3-4B-Instruct | 83.5 | 48.2 | 66.3 | 33.0 | 22.5 | 29.2 | 94 | 100 | 50 |
| InternVL3.5-38B | 64.7 | 82.2 | 62.9 | 43.4 | 26.1 | 36.9 | 96 | 100 | 50 |
| InternVL3.5-8B | 23.4 | 23.8 | 33.0 | 18.7 | 7.5 | 15.0 | 48 | 100 | 50 |
| MiniCPM-V-4.5 | 55.4 | 51.1 | 49.4 | 31.7 | 19.6 | 23.4 | 74 | 100 | 50 |
| ERNIE-4.5-VL-28B-A3B | 44.8 | 37.4 | 54.8 | 31.0 | 14.6 | 25.7 | 68 | 100 | 50 |
| Ministral-3-8B | 78.0 | 77.0 | 52.2 | 55.0 | 28.4 | 44.8 | 100 | 100 | 50 |

## 13. Closed vs open (system means)

| setting | group | 1.1 | 1.2 | 1.3 | 3.1 | 3.2 | 3.3 |
|---|---|---|---|---|---|---|---|
| Tier B r3 | closed (6) | 90.9 | 99.7 | 87.2 | 76.0 | 73.3 | 75.0 |
| Tier B r3 | open (13) | 68.6 | 77.4 | 58.7 | 53.1 | 38.6 | 46.2 |
| Tier B r3 | **gap** | **+22.3** | **+22.4** | **+28.4** | **+22.9** | **+34.7** | **+28.7** |
| Tier A r3 | closed (4) | 87.5 | 99.6 | 84.8 | 76.2 | 63.2 | 72.8 |
| Tier A r3 | open (13) | 66.7 | 89.1 | 61.1 | 60.8 | 48.7 | 55.2 |
| Tier A r3 | **gap** | **+20.8** | **+10.4** | **+23.7** | **+15.4** | **+14.5** | **+17.7** |
| Tier C r3 | closed (4) | 87.3 | 96.8 | 91.6 | 73.5 | 62.4 | 66.0 |
| Tier C r3 | open (13) | 67.0 | 62.6 | 59.2 | 42.8 | 24.2 | 36.1 |
| Tier C r3 | **gap** | **+20.2** | **+34.2** | **+32.4** | **+30.7** | **+38.2** | **+29.9** |
| attr B r1 | closed (6) | 87.0 | 99.1 | 86.2 | 76.9 | 69.5 | 70.0 |
| attr B r1 | open (13) | 64.2 | 74.0 | 52.6 | 51.7 | 35.7 | 43.5 |
| attr B r1 | **gap** | **+22.8** | **+25.1** | **+33.7** | **+25.2** | **+33.8** | **+26.5** |
| func B r1 | closed (4) | 88.4 | 98.7 | 85.8 | 75.1 | 67.2 | 68.5 |
| func B r1 | open (13) | 63.3 | 70.1 | 52.1 | 49.3 | 35.3 | 41.9 |
| func B r1 | **gap** | **+25.2** | **+28.6** | **+33.7** | **+25.8** | **+31.9** | **+26.6** |

## Level 3 change against v3.4.3 (baseline setting)

Level 1 is identical. In v3.4.3, GPT-6 Astra, GPT-5.6 Sol, Claude Opus 5 and Claude Fable 5.1 had no judge scores on
the 10 tasks added on 2026-09-17, and fell back to the computed half there. v3.4.5 uses the judge files that now
cover them.

| system | 3.1 v3.4.3 → v3.4.5 | 3.2 v3.4.3 → v3.4.5 | 3.3 v3.4.3 → v3.4.5 |
|---|---|---|---|
| **Closed-source APIs** | | | |
| GPT-6 Astra | 79.1 → 76.2 | 65.3 → 79.4 | 70.1 → 82.8 |
| GPT-5.6 Sol | 81.9 → 78.5 | 64.6 → 77.5 | 65.4 → 74.3 |
| Claude Opus 5 | 78.4 → 75.7 | 64.3 → 76.6 | 66.2 → 77.5 |
| Claude Fable 5.1 | 78.6 → 74.7 | 64.9 → 78.4 | 66.5 → 77.7 |
| Claude Sonnet 5 | 79.9 → 77.2 | 61.8 → 69.4 | 65.2 → 74.0 |
| Claude Haiku 4.5 | 76.7 → 73.3 | 57.8 → 58.4 | 60.6 → 63.4 |
| **Open-source models** | | | |
| Qwen3.5-27B | 77.9 → 70.8 | 55.9 → 53.1 | 61.1 → 64.1 |
| Qwen3.5-35B-A3B | 73.8 → 70.7 | 53.1 → 49.1 | 58.1 → 60.0 |
| gpt-oss-120B | 72.3 → 66.1 | 54.4 → 49.7 | 57.9 → 54.2 |
| Gemma-4-31B-IT | 71.0 → 64.8 | 55.0 → 52.5 | 59.6 → 61.2 |
| Qwen3-VL-32B | 51.0 → 46.9 | 40.1 → 33.5 | 43.4 → 41.5 |
| Qwen3-VL-30B-A3B | 59.0 → 52.8 | 48.9 → 37.8 | 53.2 → 47.0 |
| Qwen3-VL-8B | 43.5 → 33.4 | 38.9 → 29.1 | 39.4 → 32.7 |
| Qwen3-4B-Instruct | 55.8 → 45.1 | 45.1 → 23.5 | 49.5 → 36.1 |
| InternVL3.5-38B | 61.6 → 56.9 | 49.5 → 40.3 | 53.8 → 48.6 |
| InternVL3.5-8B | 43.8 → 34.1 | 38.1 → 30.7 | 38.4 → 32.9 |
| MiniCPM-V-4.5 | 57.5 → 46.2 | 44.2 → 31.0 | 45.2 → 33.3 |
| ERNIE-4.5-VL-28B-A3B | 45.0 → 41.2 | 38.4 → 32.0 | 40.9 → 37.7 |
| Ministral-3-8B | 64.3 → 60.9 | 51.0 → 39.6 | 55.9 → 51.5 |
| **Domain generators, image input** | | | |
| PartCrafter (K=15) | 58.6 → 62.4 | 54.9 → 50.7 | 60.3 → 64.4 |
| ↳ + Particulate | 37.0 → 30.6 | 47.5 → 33.8 | 56.1 → 54.5 |
| PartCrafter (K=8) | 62.7 → 65.0 | 54.4 → 49.6 | 59.6 → 63.6 |
| ↳ + Particulate | 41.6 → 39.2 | 51.0 → 42.7 | 57.7 → 59.7 |
| PartPacker | 55.8 → 56.7 | 60.6 → 66.1 | 62.8 → 72.0 |
| ↳ + Particulate | 46.8 → 42.2 | 58.5 → 60.4 | 60.8 → 66.8 |
| PhysX-Anything | 53.7 → 49.6 | 55.5 → 51.8 | 57.4 → 55.9 |
| **Domain generators, text input (name_only text)** | | | |
| Cube3D → CubePart | 55.2 → 57.0 | 61.3 → 68.6 | 60.3 → 63.8 |
| ↳ + Particulate | 45.3 → 43.3 | 67.5 → 85.7 | 60.3 → 64.8 |
| BrickGPT | 27.7 → 18.0 | 42.0 → 16.6 | 48.1 → 33.6 |
| LegoACE | 25.3 → 15.3 | 39.8 → 11.8 | 46.4 → 34.3 |

## Notes

- **Where the calibration was fitted.** It is fitted on 48 designs per dimension: 8 objects × 6 systems, Tier B
  round-1 checkpoints plus PartPacker and BrickGPT. All other settings (Tier A/C, other baselines, final rounds,
  ablations) are scored with the same map. Humans saw the part-coloured sheet; the judges saw clay views for 3.2
  and 3.3.
- **1.3** is `not_applicable` on tasks whose reference object is not free-standing. A single-part design is
  `not_applicable` on 1.1 and 1.2.
- **3.3:** the text-conditioned baselines never saw the photograph that 3.3 compares against.
- **Scope:** GLM-4.6V-Flash and Kimi-VL-A3B stay excluded (the owner's decision, 2026-09-21).
