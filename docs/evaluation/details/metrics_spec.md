# Metrics: from eleven dimensions to eleven algorithms

## v3.4.5 Level 3: calibration fitted on 8 objects, validated on 7 held out (2026-09-23)

This is the current Level 3 (P20). The formula is the v3.4.4 one below. What changed:
- The anchors are fitted on the **fit** objects only.
- The judge:computed ratio is **chosen** on those objects (leave-one-object-out), not fixed in advance.
- The 7 **test** objects are held out for validation.

The object split is pre-declared: alphabetical order within each of 4 categories, positions 1 and 3 to fit.

    J' = clip((judge − lo_J)/(hi_J − lo_J), 0, 1)      C' = clip((comp − lo_C)/(hi_C − lo_C), 0, 1)
    3.1 = (J' + C')/2        3.2 = J'        3.3 = (3·J' + C')/4

| dim | judge lo → hi | computed lo → hi | ratio |
|---|---|---|---|
| 3.1 | 0.361 → 0.764 | 0.128 → 1.025 | 1:1 |
| 3.2 | 0.374 → 0.725 | – | 1:0 |
| 3.3 | 0.450 → 0.777 | 0.154 → 0.606 | 3:1 |

Where things live:
- Anchors: `docs/evaluation/l3_calibration_v345.json`.
- Validation: `l3_calibration_v345_heldout.json`, written out in `l3_calibration_method.md`.
- Code: `calibrate_l3.py fit|heldout` and `spec_v345.py build`.
- Records: `results/v2/*/eval_v345/`.

On the held-out objects, v3.4.3 → v3.4.5:

| | 3.1 | 3.2 | 3.3 |
|---|---|---|---|
| MAE | 0.230 → 0.204 | 0.225 → 0.204 | 0.254 → 0.239 |
| system MAE | 0.136 → 0.098 | 0.158 → 0.148 | 0.178 → 0.146 |

## v3.4.4 Level 3: calibrated to the human study (2026-09-23)

Requested by the owner after the Level 3 human study (`human_evaluation/`, P19 in `metrics_feedback.md`).
Implementation:
- `ppbench/v2/calibrate_l3.py` fits the anchors and writes `docs/evaluation/l3_calibration_v344.json`.
- `ppbench/v2/spec_v344.py` writes `results/v2/*/eval_v344/`.

Levels 1, 2 and 4 are unchanged. The two raw halves of each Level 3 dimension (judge and computed) are exactly the
v3.4.3 ones. Only the step from halves to score changes: each half is put on the human scale, and then the halves are
blended with an integer ratio.

**Step 1: raw halves (unchanged from v3.4.3).** For a design x:

    judge(x)    = mean over judges m in {Qwen2.5-VL-32B, gemma-3-27b} of
                  mean over the dimension's specific criteria c of (s_{m,c} - 1) / 9,     s in 1..10
                  (3.1 prompt v1: boundaries, granularity, separability;
                   3.2: proportion, coherence, refinement;
                   3.3: inventory, placement, relative_scale, detail_fidelity)

    comp_3.1(x) = mean of the available terms among
                  granularity    = 2^-|log2(n_parts / n_ref_parts)|        (segmented reference only)
                  consolidation  = 1 - violations / (n_parts - 1)           (None if the design declares no materials)
                  joint_place    = 1 - rejected / declared joint pairs      (None if no joints are declared)
    comp_3.2(x) = none                                                      (P15)
    comp_3.3(x) = mean(part_iou, symmetry_match)
                  part_iou       = sum_r vol_r * max_p IoU(r, p) / sum_r vol_r   (reference parts r, design parts p, 1 cm grid)
                  symmetry_match = 1 - |sym - sym_ref| / max(sym_ref, 1 - sym_ref)

**Step 2: calibration to the human scale.** For a half h with anchors (lo_h, hi_h):

    h'(x) = clip( (h(x) - lo_h) / (hi_h - lo_h), 0, 1 )

The anchors are fitted per dimension and per half on the human-rated items. These are 90 designs per dimension: 6
systems x 15 objects, Tier B round-1 checkpoints plus the PartPacker and BrickGPT outputs. The human target is
H = mean over raters of (r - 1) / 4, with r in 1..5. The anchors are chosen so that on these items h' has the human
mean and standard deviation:

    b  = sd(H) / sd(h),   a = mean(H) - b * mean(h)
    lo = -a / b,          hi = (1 - a) / b

This is moment matching. Least squares was tried and rejected: it shrinks the spread back towards the middle (item
sd 0.18 against the human 0.34) and has twice the system-level error.

| dim | half | lo | hi | fitted on |
|---|---|---|---|---|
| 3.1 | judge | 0.3656 | 0.7559 | 90 items |
| 3.1 | computed | 0.1428 | 1.0401 | 84 items |
| 3.2 | judge | 0.3715 | 0.7434 | 90 items |
| 3.3 | judge | 0.4708 | 0.7857 | 90 items |
| 3.3 | computed | 0.2060 | 0.6579 | 72 items |

In words: a raw 3.2 judge score of 0.372 (about 4.3/10) maps to 0, and 0.743 (about 7.7/10) maps to 100. The
exact values are in `l3_calibration_v344.json`, whose sha is stored in every record and in its stamp, so a refit
rebuilds the records.

**Step 3: blend with an integer ratio.**

    3.1 decomposition        S = (1 * J' + 1 * C') / 2
    3.2 aesthetics           S = J'
    3.3 structure alignment  S = (3 * J' + 1 * C') / 4

The ratios were chosen by leave-one-task-out cross-validation over 1:0, 4:1, 3:1, 2:1, 3:2, 1:1, 2:3, 1:2, 1:3 and
0:1 (`calibrate_l3.py cv`). 1:1 has the lowest error for 3.1, and 3:1 is at the minimum for 3.3. The error curve is
flat between 2:1 and 1:2, so the simplest ratio at the minimum was taken.

**Missing halves and failures (as in P13).**
- One half missing: the score is the other calibrated half. Status is `pass` when the judge is present and
  `degraded` when only the computed half is left.
- Neither half: `skipped`.
- An empty or unloadable design stays `fail` with score 0.
- System scores are the mean over tasks (x100). A `not_applicable` or `skipped` dimension is left out of the mean.

**Agreement with humans (leave-one-task-out, 90 items per dim).**

| | 3.1 | 3.2 | 3.3 |
|---|---|---|---|
| MAE vs human, v3.4.3 → v3.4.4 | 0.217 → 0.208 | 0.236 → 0.217 | 0.259 → 0.215 |
| item Spearman, v3.4.3 → v3.4.4 | 0.48 → 0.49 | 0.64 → 0.59 | 0.64 → 0.66 |
| system-level MAE (6 systems) | 0.076 | 0.082 | 0.097 |
| inter-annotator Spearman (30 double-rated items) | 0.69 | 0.52 | 0.73 |

## v3.4 Level 3 (2026-09-20)

Requested in the Level-3 review logged in `docs/evaluation/metrics_feedback.md` (P8-P13);
implemented in `ppbench/v2/spec_v34.py`, with the renders in `judge_views.py` and the judge in
`judge_l3.py`. Levels 1, 2 and 4 are carried through from the v3.3 record unchanged. Level 3 goes
from two dimensions to three, and the VLM judge becomes the primary evidence in all three.

    3.1 decomposition        0.30 judge + 0.70 computed (v3.4.2, owner) -- judge prompt v1
    3.2 aesthetics           1.00 judge (v3.4.1)            no reference anywhere: no photo, no computed term
    3.3 structure alignment  0.70 judge + 0.30 computed     new dimension

**What the judge sees.** Three calls per design, because the three questions need different
pictures and a request is capped at two images.

| dimension | image 1 | image 2 | criteria (1-10, written anchors) |
|---|---|---|---|
| 3.1 | the condition photograph, for context | `<key>_decomp.png`: two part-coloured views over two exploded views, one hue per part | **prompt v2 (v3.4.1):** over_split, fused, boundaries, *overall* — after a four-step roadmap (list components, check each one's colours, check moving components, score) and an anti-constant calibration paragraph |
| 3.2 | `<key>_claygrid.png`: four neutral-clay views | — | proportion, coherence, refinement, *overall* |
| 3.3 | the condition photograph | `<key>_claygrid.png` | inventory, placement, relative_scale, detail_fidelity, *overall* |

Renders are featureless grey clay for 3.2 and 3.3 so a declared palette cannot buy an aesthetics
point and so the alignment question is about structure, not finish. The decomposition sheet is the
opposite: one hue per part is the entire content. The judge is never shown a part list, coordinates
or the identity of the system, and 3.1's prompt carries only the part *count* as text.

A dimension's judge score is the mean of its **specific** criteria; `overall` is collected as an
independent holistic cross-check and deliberately not folded in twice. Raw 1-10 answers are
normalised `(s-1)/9`, temperature 0.

**Judges.** Two models that are not systems under test: `Qwen2.5-VL-32B-Instruct` and
`gemma-3-27b-it` (the contestants are Qwen3-VL-\* and gemma-4-31B-it). Each scores every design
independently; `judge_l3.py agreement` reports the Spearman rank correlation between them per
dimension, and that figure is published next to the scores. The reference design is scored like a
submission and is the calibration anchor. A human calibration subset is still owed.

**3.1 computed third.** `mean(granularity, consolidation, joint placement)` over whatever is
available.
`granularity = 2^-|log2(n_parts / n_reference_parts)|` (needs a segmented reference: 40 of 50 tasks).
`consolidation = 1 - violations / (n_parts - 1)`, a violation being two bodies rigidly connected, of
the same material, that never move relative to each other.
`joint placement = 1 - rejected / declared joint pairs`, a joint being rejected when its origin is
more than 2 cm from both surfaces. It is **one-sided on purpose**: protocol v2.2 asks for a joint
wherever the object *moves*, not at every seam, so "every boundary carries a joint" would grade
against a rule nobody was given. A design that declares no joints scores `None` here, not 1.0 —
silence must not buy a point, and the absence is already charged in 1.1 and 2.3.

**3.2 computed half — removed in v3.4.1 (P15).** 3.2 is now 1.00 judge. The former term, kept in the record for information only, was `proportion_score`: `2^-mean(|log2 (H/W) / (H/W)_ref|,
|log2 (D/W) / (D/W)_ref|)`, proportions against the category prior. This is the only rule term in
the old 3.2 that was not backwards.

**3.3 computed third.** `mean(part_correspondence_iou, symmetry_match)`.
`part_correspondence_iou` moved here from 3.1 (P10): for each reference part, the best IoU against
any design part on the scale-normalised 1 cm grid, weighted by reference part volume — it measures
"is the reference's structure where the reference put it", which was never a decomposition term.
`symmetry_match = 1 - |sym - sym_ref| / max(sym_ref, 1 - sym_ref)` (P11). Absolute mirror symmetry
was a **backwards** quality signal: the 40 reference assets average 0.803 mirror IoU, and
Qwen3-4B-Instruct averaged 0.948 and beat its own task's reference on 79 % of tasks. Matching a real
product's asymmetry now scores 1.

**3.3 against 2.1.** 2.1 geometry compares meshes: overall shape, scale and silhouette against the
reference asset. 3.3 compares the design to the **photograph** and asks the fine-grained questions a
mesh metric averages away — the count of repeated elements, the cross-section of members, cut-outs,
which part is where relative to which. A design can pass 2.1 on silhouette and fail 3.3 on having
three legs where the photograph has five.

**Missing halves (P13).** `0.7 judge + 0.3 computed` applies when both halves exist. Judge only →
scored on the judge, status `pass`, note recorded. Computed only (the judge did not return parseable
JSON) → status `degraded`. Neither → `skipped`. An empty or unloadable design stays `fail`
regardless of what the judge said. Coverage of each half is reported per system.

## v3.3 Level 1 (2026-09-20)

Requested in the Level-1 review logged in `docs/evaluation/metrics_feedback.md`; implemented in
`ppbench/v2/spec_v33.py`, which reloads and re-voxelises every design (v3.2 only re-partitioned
stored numbers). Levels 2-4 are carried through from the v3.2 record unchanged.

**1.1 connectivity.** Two questions in one dimension: does the assembly hold together, and does the
design say *how*. The graph is the rule protocol v2.2 promised the agents — surfaces touching within
about 4 mm, or a declared joint whose origin is within 2 cm of both surfaces — so nothing is graded
against a rule it was never given. The fastening question is asked as a credit instead: a pair is
*fastened* when the eroded cores overlap by >= 0.1 cm3 (*interlock*), or a declared joint sits on a
real interface >= 1 cm2 (*fastened joint*), or a snap-fit lattice part rests on the one below within
1.5 mm (*lattice stud*). A contact with none of these is a *resting contact*: it still holds the
graph together, it just earns no fastening credit.

    1.1 = mean(anchored_volume_frac, break_score, fastening_spec_rate)      one third each
    break_score         = 1 - (n_components - 1) / (n_parts - 1)
    fastening_spec_rate = min(1, n_fastened_connections / (n_parts - 1))

`n-1` is what a body of `n` parts needs to span, so the third term is the share of the minimum
structure the design actually specifies. Declaring joints raises it; more parts do not lower it.
`whole_body_rate` (1.0 in 92% of records) and `fastening_strict_score` (1.1 with resting contacts
holding nothing) are reported, not scored. A one-part design is `not_applicable`: it has no
connection to evaluate and must not collect 1.0.

**1.2 collision.** Every interfacing pair is in the denominator now, declared joints included; a
declared joint widens the tolerance from 10% to 25% of the smaller part instead of exempting the pair.
A design with no interfacing pair at all is `not_applicable`, not 1.0.

**1.3 stability.** Two equally weighted questions, on the whole design as placed (all parts, the
ground plane the design itself rests on -- not the largest connected group, which would make 1.3 a
second reading of 1.1 and, when that group floats, take its own lowest rim for the ground):

    1.3 = mean(stand_score, push_score)
    stand_score = clip(critical_tilt, 0, target_tilt) / target_tilt
    push_score  = clip(f / target_f, 0, 1),  f = support_margin / (z_top - z_ground)

`f` is the horizontal force, in multiples of the assembly's own weight, needed at its highest point to
tip it about its weakest support edge. Since `tan(tilt) = margin / CoM height`, `f` is tilt times the
CoM-height-to-total-height ratio: the push term is what puts top-heaviness into the dimension.
An open mesh whose voxel fill failed is re-weighted by its own convex hull, so it stops weighing
nothing and silently moving the centre of mass.

## v3.2 layout (2026-09-20)

The dimension ids below are the v3.1 ones. v3.2 re-partitions them; `ppbench/v2/spec_v32.py`
derives a v3.2 record from a stored v3.1 record, so nothing here is re-simulated.

| v3.2 | name | comes from |
|---|---|---|
| 1.1 | connectivity | v3.1 1.1, the component / whole-body terms |
| 1.2 | collision | v3.1 1.1, the collision-free pair rate (penetration volume reported, not yet scored) |
| 1.3 | stability | v3.1 1.2 |
| 2.1 | geometry | v3.1 2.1 |
| 2.2 | part-level alignment | **new**: are the parts the claims require there at all, and connected |
| 2.3 | kinematics | v3.1 2.2 |
| 3.1 / 3.2 | decomposition / aesthetics | unchanged |
| 4.1 / 4.2 | material / assembly sequence | unchanged |
| 4.3 | operability | v3.1 2.3 |

v3.1 4.3 (functional completeness) stops being a dimension: its part-presence half is what 2.2 now
scores, its connection half is a reported metric of 2.2.

**Role-dependent dimensions are skipped, not zeroed, on an output that carries no role** (the mesh
generators): 2.2 and 4.3 return `None` there, the way 4.1 already did for a design with no material.
A zero would read as "unusable" when the truth is "not expressible".


The root README says what each dimension asks. This says how each one is
computed, what it returns, and which layer of the environment produces it.

Two rules run through all of it.

**Prefer graded over binary.** Penetration volume beats "collides". Critical
tilt angle beats "stands". Achievable range ratio beats "the door opens". A
binary metric saturates, hides progress, and cannot be regressed against a human
reference. `BrickForge`'s own tolerance sweep is the argument: at 100 LDU³ of
allowed interpenetration, one model moves 3.9 points and another moves zero, and
a binary check would have shown neither.

**One primitive, many metrics.** Voxelize the assembly once per scene, at 4 LDU
by default and 1 LDU on demand. Interpenetration, void detection, clearance,
reach, passability, swept volume, and symmetry are all operations on that grid.
Building seven bespoke geometry pipelines is how this project would drown.

---

## Level 1 — Structure

### 1.1 Integrity · Computed

| Check | Algorithm | Returns |
|-------|-----------|---------|
| Well-formed | catalog parts are valid by construction, so verify resolution instead: every instance maps to a catalog id at a known pose. Free-form mesh output goes through `manifold3d` for watertight / manifold / normal consistency. | `resolve_rate`, `manifold_rate` |
| No interpenetration | BVH broad phase on part OBBs, exact triangle intersection narrow phase, then **volume** of overlap by local voxel intersection | `pen_volume_ldu3` per pair, p50 / p95 / max, and the rate at a swept tolerance |
| Connected | connected components of the contact graph | `n_components`, `largest_component_frac`, `n_floaters` |
| Whole bodies | each declared body is one connected component **and** internally rigid (no declared joint inside it) | `whole_body_rate`, list of violations |

Report the tolerance sweep, not one threshold. Carry over the finding that
port-conflict overlaps and no-shared-connector overlaps respond differently to
tolerance: keep the two classes separate in the output, because that split is
where the interesting signal was.

### 1.2 Stability · Computed, then Simulated

Three metrics, increasing in cost.

**Support margin (Computed).** Contact points with the ground plane, convex
hull, distance from the CoM projection to the hull boundary, normalized by the
hull's inradius. Continuous, instant, and it is the textbook definition.

**Critical tilt angle (Computed).** Rotate gravity in the plane and bisect for
the angle at which the CoM projection leaves the support polygon. Sweep the
azimuth and report the minimum over directions. This is the best headline
stability number available: continuous, bounded, physically meaningful,
comparable across every design in the suite, and independent of any solver.

**Settling displacement (Simulated).** MuJoCo, gravity on, N seeds with small
initial perturbations, T seconds. Report max part displacement and CoM drift as
a median with spread across seeds. Uniform density per the level's scope. This
metric's job is to catch what statics cannot — a design that is in equilibrium
but only just, or one that shakes itself apart — and to validate the LP on a
sample. If the seed spread is wide, it stays a diagnostic and does not enter the
headline.

MJX makes the seed sweep nearly free on the available GPUs, which is the reason
to insist on reporting spread rather than a single rollout.

---

## Level 2 — Affordance

### 2.1 Geometry · Computed

A predicate library keyed on `role`, each returning a continuous residual. These
are the ones worth writing first:

- **wheel** — fit a surface of revolution to the body's points; return radial
  residual as a fraction of radius, plus the fitted axis
- **seat / shelf** — largest horizontal planar region by area, its height above
  ground, and its aspect; a seat is a flat region of adequate area in a height
  band
- **container** — flood-fill the voxel grid from outside; enclosed void volume
  is the capacity, and the opening is the boundary between void and outside
- **blade / edge** — minimum local thickness from the distance transform along
  the edge curve
- **handle / rung** — a graspable feature is one with free space of finger
  radius on two opposing sides

**Size, scored separately.** `log2(bbox_extent / category_prior)`, per axis and
on the diagonal. Report the log error, the fraction within a factor of two, and
the sign of the bias. Absolute scale is the most common failure in generated
geometry and it deserves its own column rather than being folded into a
geometry score.

### 2.2 Kinematics · Computed

Compile the declared joints to MJCF, then:

- **Declaration completeness.** Type, axis, and range present for every joint.
  Missing means zero, per fail-closed.
- **Realization consistency.** The connector family that realizes the joint must
  be able to produce the declared DOF. A revolute joint realized by a stud
  field is a welded joint mis-declared. The catalog's five closed joint families
  make this a lookup, not a judgment.
- **Achievable range ratio.** Sweep each joint across its declared range in K
  steps, collision-check at each. Return the fraction of the declared range
  reachable without collision, and the first blocking angle. This is the metric
  that operationalizes "the door sweeps its arc without hitting the fender," and
  being a ratio rather than a bit it distinguishes a door that opens 15 degrees
  from one that does not open at all.
- **DOF match.** Declared mobility against the task template's expectation.

For coupled mechanisms, sample the reachable configuration space rather than
sweeping joints independently, and say in the report which you did.

### 2.3 Operability · Computed

All three of the root README's clearances reduce to free-space queries on the
voxel grid, which is why this stays Computed even though it introduces an
external body.

- **Room to grip.** A finger probe of standard radius must fit in the gap
  between the handle and the body, on two opposing sides. Return the maximum
  inscribed probe radius.
- **Room to reach.** Shortest free path from the declared `stand_at` / `eye`
  frame to each control, on the free-space voxel graph, with the path's minimum
  clearance as the score. A control the arm cannot get to is unreachable even if
  it is geometrically nearby.
- **Room to pass.** Maximum inscribed radius along the free path through the
  opening. One number, and it directly answers whether a person or a payload
  fits.
- Recompute at each sampled configuration from 2.2, and report the worst case.
  A door that only blocks the handle when half open is still a broken door.

---

## Level 3 — Design

Adjudicated, with computed components pulled out wherever they exist. The
discipline here is what makes the level survive review.

**Judge on renders plus structured facts, never on a part list.** Turntable,
section, and exploded views, alongside part count, symmetry residual, material
set, and body graph. A judge reading raw coordinates is guessing.

**Pairwise, not absolute.** Ask which of two designs is better on one criterion.
Pairwise comparison is far more reliable than a 1-5 scale, and it aggregates to
a rating. Anchor every comparison set with reference designs so the scale is
tied to something real.

**Report agreement before reporting scores.** Human agreement on a calibration
set, and judge-versus-human agreement, both stated. A Level 3 number without an
agreement figure next to it is not evidence.

### 3.1 Decomposition — partly Computed
Computed: part count against the reference distribution for the category;
consolidation violations (two bodies, rigidly connected, same material, never
moving relative to each other); standard-vocabulary fraction; boundary-function
alignment (does every declared joint sit on a body boundary, and does every body
boundary carry a joint). Adjudicated: whether the cuts make sense.

### 3.2 Aesthetics — partly Computed
Computed: best reflective symmetry plane and its residual on the voxel grid;
proportion against the category prior; a boolean-scar proxy for free-form
geometry. Adjudicated: style coherence across parts, which is the part no
formula reaches.

### 3.3 Creativity — partly Computed
Computed: novelty as distance to the nearest reference design in the BrickForge
embedding space, and diversity as mean pairwise distance across repeated samples
from the same prompt, plus part-vocabulary entropy. Adjudicated: whether the
novelty is a solution or noise. Grounding creativity in a retrieval distance is
what stops this dimension from being pure vibes, and the memorization check —
whether repeated sampling returns one answer — is worth reporting on its own.

---

## Level 4 — Realization

### 4.1 Material · Computed, then Simulated

- **Resolution.** Every body's material resolves to a row with density,
  stiffness, and friction. Coverage is a number; a design with unassigned
  materials is skipped here rather than scored zero.
- **Compatibility.** A role-material table: transparent where sight is required,
  compliant where a seal is required, and so on. Violations are enumerated.
- **Re-run statics with real properties.** The LP from 1.2, now with true masses
  and per-joint capacities, returns a **safety factor per joint** and names the
  minimum. This is the dimension where the LP pays for itself: the shelf that
  stood under uniform density and sags once it is a real material shows up as a
  safety factor below one at a named joint.
- **Deflection (Simulated, cheap).** Each brick a node, each joint a 6-DOF
  spring at measured stiffness, one linear solve. Returns max deflection under
  the load case. Far cheaper than FEM and adequate at this scope.

Joint capacity and stiffness numbers must be measured or cited, with a pointer,
and marked as placeholders until they are. Following the house rule from
`representation_design.md`: no invented numbers.

### 4.2 Assembly Sequence · Computed

The cleanest dimension in the benchmark, because disassembly planning is exact
here. Plan the removal and reverse it.

- **Precedence.** Walk the declared order. At each step, does the part being
  added have a collision-free insertion path given only the parts already
  placed? For bricks the path is almost always along the connector axis, so test
  the connector axis plus the six axis-aligned directions as an infinite
  translation. Returns the fraction of steps that are feasible and the index of
  the first violation.
- **Intermediate stability.** Run 1.2 on every prefix of the sequence. Returns
  the minimum support margin over the build. This catches the design that is
  stable when finished and falls over at step 12, which is a real
  buildability failure that a final-state-only benchmark cannot see.
- **Hand count.** Number of disconnected components at each prefix. More than
  one means the builder is holding a loose piece while placing another, which is
  the "third hand" failure stated as a graph property.
- **Order quality.** Compare against the optimal order found by search, to
  separate "no valid order exists" from "the agent chose a bad one."

### 4.3 Functional Completeness · Computed at the interface, Declared inside

Per the scoping rule. For each subsystem the task's purpose requires:

- **Presence.** Is a body declared for it?
- **Placement.** Does it satisfy the category's region predicates — the engine
  inside the engine-bay box, the wheel below the chassis, the control within the
  reach envelope from 2.3?
- **Connection.** Is there a path in the part graph from the power source to
  each actuated body, through edges whose joint family can transmit the required
  motion? A stud connection cannot transmit rotation; an axle can. For geared
  paths, the ratio is a symbolic product along the path.

Return presence, placement, and connection as three separate coverage numbers.
Never merge them into a single "function" score, and never let anything from
`claims` enter any of the three.

**Then a Simulated tier, run only when the computed tier passes.** Existence of
a torque path is not the same claim as the thing moving. Rolling (slip ratio,
climbable grade, fender rub under load), driving (distance, heading drift,
turning radius), mechanism throughput (measured gear ratio against the symbolic
prediction, backlash, binding), and load holding (deflection, joint separation).
Specified in `../environment/simulation.md` §4, with the engine choice and the
cross-engine agreement requirement in §2 and §3. Internal subsystem physics
stays Declared.

---

## The relevance gate

Run before the levels. A VLM on turntable renders plus a category check, with
its own accuracy measured on a labeled set. Returns pass or fail. A perfect
table submitted for a chair task is a failed task, not a well-formed one, and
the gate's own error rate belongs in the report.

---

## Report shape

One record per design, and never a single averaged score.

```jsonc
{
  "task_id": "...", "source": "brickgpt", "seed": 0,
  "versions": {"catalog": "sha...", "code": "sha...", "mujoco": "...", "bpy": "..."},
  "ingest": {"parse_ok": true, "resolve_rate": 1.0, "repair_ldu": 0.4},
  "gate": {"pass": true, "gate_confidence": 0.91},
  "dims": {
    "1.1": {"depth": "computed", "status": "pass",
            "pen_volume_p95": 0.0, "n_floaters": 0, "whole_body_rate": 1.0},
    "1.2": {"depth": "computed", "status": "pass",
            "support_margin": 0.31, "critical_tilt_deg": 22.4,
            "settle_disp_mm": {"p50": 0.2, "spread": 0.05}},
    "2.2": {"depth": "computed", "status": "fail",
            "achievable_range_ratio": 0.18, "first_block_deg": 32}
  },
  "budget": {"verify_calls": 0, "tool_calls": 0, "wall_s": 4.1},
  "headline": {"gate_pass": true, "critical_fail": ["2.2"], "buildable": false}
}
```

The single headline number stays what the root README already defines: the
fraction of designs that pass the gate and fail nothing critical on levels 1, 2
and 4. Everything else is reported per dimension, with its depth label and, for
Simulated entries, its spread.
