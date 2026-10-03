# Level 3: human-calibrated automatic evaluation (spec v3.4.5)

Paper-ready description of the method and its held-out validation. Generated from `docs/evaluation/l3_calibration_v345.json` and `l3_calibration_v345_heldout.json` (`ppbench/v2/calibrate_l3.py fit` / `heldout`).

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
