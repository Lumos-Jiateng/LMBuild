# Level 2 (Affordance): 2.1 Geometry, 2.2 Parts, 2.3 Kinematics (spec v3.7, 2026-09-22 evening)

**Status: proposed.** The author has not yet adopted it, and it supersedes the v3.6 proposal
(`results_level2_v36.md`, kept for history). Where things are:
- Records: `results/v2/*/eval_v37/`. Sheets: `results/v2/*/afford/sheet_v37.json`.
- Code: `ppbench/v2/afford/rules.py`, with joint quality in `afford/kin.py`.
- Golden scenes: `python -m ppbench.v2.afford.golden`. All pass, including the 15 checks added for v3.6 and v3.7.

Rebuild with `python -m ppbench.v2.afford.rules sheets && python -m ppbench.v2.afford.rules score --force`. It needs the
`.venv_eval` python and takes about 13 minutes on 110 workers for 3,038 designs: Tier A, B and C, plus the baselines.
No VLM or LLM is called anywhere in Level 2.

## Changes from v3.6, at the author's request

1. **2.1 is geometry only.** Creation is reported but never scored. Retrieving catalogue parts costs nothing, and
   creating good parts costs nothing. Tier C (creation only) is scored with the same rule.
2. **2.3 is tied to specific joints.**
   - The motion targets are the reference object's own annotated joints (Artiverse, and the LEGO references'
     connector joints) plus the claimed motions of visible parts.
   - Every joint the design declares is checked for being reasonable. 2.3 is the F1 of target recall and joint
     precision.
3. **2.2 is easier on the design.**
   - A part that is there but not named is found by where it lies against the reference.
   - An internal module that is not named is found by looking inside: no air path from outside reaches it.

## How each number is computed

### Shared

| step | rule | why |
|---|---|---|
| claims | The current verified claims in `core_v2.json` (P, F, K), not the frozen run snapshot. | The agents never saw the claims (name_only+image). The snapshot still held claims the 09-20 re-verification dropped, which zeroed whole tasks. |
| components | One per claimed part phrase, merged through the task lexicon. Frame words (`ground`, `floor`, the object's own name) are dropped. | These are the parts the object's documentation says it needs. |
| naming a part | The design's declared role through the Grounder (details in `rules.py`). | The design's own statement is the evidence. Nothing is guessed, and every rule is linguistic, not per object. The rules are listed under this table. |
| placing a part | A part whose role matches nothing, or every part of a design with no roles, is matched by geometry. It counts for a component when at least half its surface lies within 5% of the diagonal of the reference's parts of that component, after scale normalisation and best yaw. | A part's presence should not depend on its name. |
| instances | Parts of the same component joined by a fixed joint, or with overlapping boxes, are one instance. | A tire on its rim is one wheel; four legs fixed to one seat are four legs. |
| counts | Required only for plural claim phrases. The count is the 25th percentile over the real instances that model the component, capped at 8. | "Wheels" means several. "Keyboard" means one, even though datasets label every key. |
| real instances | The reference plus up to 11 extended-set objects of the same Wikidata class and medium, grounded by their dataset labels only. | "Right" is what real objects do, not a hand rule. |
| visible | The reference models the component, or at least 40% of the instances do. | Otherwise the component is internal: it is judged on declaration, or on an unnamed interior module. |

The Grounder rules, in order:
1. Contiguous token runs of the component phrase, spaced and joined.
2. When several match, the one that ends last wins (the head noun).
3. The tail of a phrase (`spout` matches `outlet spout`).
4. Comma conjuncts in order (`door panel, handle`), then the bracketed text.
5. Lexicon synonyms.
6. `<X> <WordNet part of X>` (`wheel rim`).
7. The modifier of an abstract head (`steering system` is matched by the `steering column`).
8. British and American spellings are treated as one (`tyre`/`tire`, `castor`/`caster`).

### 2.1 Geometry = 0.6 shape + 0.4 part geometry

| term | rule | why |
|---|---|---|
| shape | F-score at 2.5% of the diagonal between 20k surface samples of the design and of the reference. Each is centred on its footprint and divided by its own diagonal, and the best of 4 yaws is taken. The 10 LDraw-reference tasks are included. | This is the reviewed v3.2 metric. Size is excluded because only proportions are comparable across units. |
| part geometry | For each visible component the reference has: the F-score at 5% between the design's parts *declared* as it and the reference's parts of it, in the same frame. The mean is over components, and a missing component scores 0. | It asks whether the wheels are wheel-shaped and sit where wheels sit. It uses declared parts only: a part *found* by where it lies would be compared with the very region that found it. |
| role-less designs | Shape only. | Part geometry is not expressible for them. |
| creation | Reported, never scored: the created share of the surface, and the precision and recall of the created geometry. | A catalogue part placed right is right. A created part is judged by the same geometry and neither gains nor loses for being created. |

### 2.2 Parts = 2/3 visible + 1/3 internal

| term | rule | why |
|---|---|---|
| component score | min(1, instances / required count). An instance floating off the assembly counts one half, using the Level 1 `unattached_parts`. | A part must be there, as many as the object needs, and attached. |
| visible components | Found by name, or by placement against the reference. | |
| internal components | Found by name. Otherwise each **unnamed interior module** counts 0.5 for one unmet internal component. An unnamed interior module is a group of unnamed parts at most 5% of whose boundary cells touch the outside air (voxel flood fill from the grid border). | Not declaring something does not prove it is absent. A sealed module inside the housing is evidence of an internal part, but its identity is unstated, so it earns half. |
| brick fill | Interior LDraw bricks never count as modules. | A brick model's hidden bricks are structural fill, not an engine. |

### 2.3 Kinematics = F1(target recall, joint precision)

| term | rule | why |
|---|---|---|
| targets | (a) The reference's annotated joints on Artiverse (curated articulations) and LEGO (connector joints) references, grouped by moving component and motion type. The number of slots is how many such joints the reference has, capped at 8. (b) Claimed motions of visible parts. A claim merges with (a) when it names the same component and motion. When the reference's labels ground under 30% of its parts (LEGO bricks, some curated CAD), visibility is unknown and every declarable claim is a target. | These are the motions this object and its documentation say it has. Internal-mechanism claims (pistons, swashplates, fan blades inside a laptop) are not required. Fusion 360 joints are excluded because they are CAD mates, mostly screws. |
| expectation | The ground truth first: the reference's own joint relation to its part. Otherwise at least 2 real instances. Otherwise the generic rule (type, range, sweep, moveset, plus the symmetry axis through the centre for a round part turning on an axle). | "Evaluate against the ground truth if possible." A single extended asset is its own annotation, not the category. |
| q(joint, target) | type_ok × axis_ok × offset_ok × range_ok × free_ok × moveset_ok (defined in `afford/kin.py`). | A joint must be on the right part, about the right axis, in the right place, over the right range, move freely, and not drag what stays. |
| recall | For each target, the mean over its slots of the best q among the joints that move an instance of its component. The mean is then taken over targets. | Did the design realise the object's motions? |
| precision | For each declared moving joint, the best q it earns on any target or internal claim. A joint on no motion gets 0.5 × its soundness as a joint: moves a proper sub-assembly, freely, over 30 deg or 20% travel. | Are the declared joints reasonable? A sound unclaimed joint (a lever handle) keeps half, and junk joints pull the score down. |
| F1 | 2RP/(R+P). | Without any realised target a design scores 0, however many sound joints it has. Under a weighted mean, Particulate's generic joints earned 14–25. |

## Results (main setting, final design, x100; N = tasks)


### Main table — Tier B (catalogue + created parts) and domain baselines (30 rows)

| system | 2.1 | shape | part geo. | 2.2 | visible | internal | 2.3 | recall | precision | v3.2 2.1 / 2.2 / 2.3 | N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *Closed-source APIs* | | | | | | | | | | | |
| GPT-6 Astra | **76.7** | 83.4 | 66.7 | **65.2** | 86.5 | 19.3 | **54.8** | 51.5 | 71.5 | 84.4 / 57.4 / 92.1 | 40 |
| GPT-5.6 Sol | **62.6** | 68.2 | 54.2 | **64.5** | 85.3 | 19.8 | **52.6** | 47.0 | 72.6 | 68.7 / 56.0 / 92.9 | 40 |
| Claude Opus 5 | **65.4** | 71.0 | 56.9 | **65.3** | 83.8 | 26.2 | **48.2** | 43.1 | 66.4 | 70.9 / 60.3 / 90.8 | 40 |
| Claude Fable 5.1 | **65.9** | 69.6 | 60.3 | **63.5** | 85.2 | 17.6 | **54.1** | 47.3 | 76.6 | 69.6 / 56.8 / 92.8 | 40 |
| Claude Sonnet 5 | **55.7** | 61.2 | 47.4 | **61.6** | 82.8 | 20.2 | **43.6** | 39.8 | 66.2 | 62.6 / 52.6 / 93.9 | 50 |
| Claude Haiku 4.5 | **39.4** | 45.9 | 29.5 | **58.4** | 80.5 | 13.6 | **23.3** | 18.7 | 55.3 | 46.1 / 43.2 / 83.9 | 50 |
| *Open-source models* | | | | | | | | | | | |
| Qwen3.5-27B | **46.3** | 52.8 | 36.7 | **53.0** | 72.4 | 14.6 | **20.8** | 16.5 | 51.4 | 53.9 / 43.4 / 84.4 | 50 |
| Qwen3.5-35B-A3B | **36.9** | 45.3 | 27.4 | **50.3** | 74.3 | 7.7 | **17.6** | 14.8 | 51.8 | 43.4 / 41.5 / 86.4 | 50 |
| gpt-oss-120B | **31.2** | 37.0 | 22.5 | **50.2** | 69.7 | 10.6 | **16.7** | 12.6 | 45.6 | 38.4 / 42.9 / 79.1 | 50 |
| Gemma-4-31B-IT | **41.1** | 47.1 | 34.1 | **46.9** | 69.3 | 4.9 | **17.3** | 13.6 | 48.8 | 47.1 / 40.7 / 79.6 | 50 |
| Qwen3-VL-32B | **23.3** | 38.3 | 18.9 | **36.8** | 67.7 | 8.9 | **7.4** | 7.7 | 32.1 | 29.2 / 31.7 / 52.8 | 50 |
| Qwen3-VL-30B-A3B | **24.5** | 32.0 | 15.5 | **41.7** | 61.9 | 5.6 | **3.6** | 2.8 | 10.9 | 31.2 / 36.6 / 31.3 | 50 |
| Qwen3-VL-8B | **16.4** | 26.6 | 12.4 | **30.8** | 56.0 | 5.5 | **4.8** | 4.6 | 27.5 | 19.1 / 27.9 / 58.2 | 50 |
| Qwen3-4B-Instruct | **23.4** | 29.8 | 12.8 | **38.2** | 53.5 | 6.6 | **3.9** | 2.8 | 20.8 | 29.2 / 35.4 / 65.4 | 50 |
| InternVL3.5-38B | **24.1** | 32.0 | 13.6 | **33.4** | 48.9 | 4.9 | **9.9** | 7.8 | 29.3 | 32.0 / 27.8 / 59.1 | 50 |
| InternVL3.5-8B | **15.5** | 27.8 | 9.9 | **19.4** | 35.6 | 5.0 | **2.0** | 1.7 | 11.2 | 20.0 / 18.6 / 20.9 | 50 |
| MiniCPM-V-4.5 | **20.6** | 26.8 | 15.1 | **28.0** | 43.1 | 3.3 | **2.3** | 1.7 | 14.8 | 26.3 / 23.0 / 35.5 | 50 |
| ERNIE-4.5-VL-28B-A3B | **17.4** | 27.7 | 14.7 | **28.7** | 55.2 | 4.5 | **4.0** | 4.0 | 22.4 | 20.9 / 24.2 / 35.4 | 50 |
| Ministral-3-8B | **23.2** | 28.7 | 14.9 | **46.4** | 65.2 | 7.9 | **5.5** | 3.6 | 25.1 | 27.4 / 40.1 / 53.8 | 50 |

| PartCrafter (K=15) | **45.8** | 45.8 | nan | **27.3** | 41.0 | 0.3 | **0.0** | 0.0 | 0.0 | 43.0 / nan / 0.0 | 50 |
| ↳ PartCrafter K=15 + Particulate | **45.8** | 45.8 | nan | **11.0** | 16.2 | 0.0 | **2.5** | 1.8 | 35.0 | 42.3 / nan / 81.7 | 50 |
| PartCrafter (K=8) | **41.3** | 41.3 | nan | **20.8** | 31.3 | 0.1 | **0.0** | 0.0 | 0.0 | 36.7 / nan / 0.0 | 50 |
| ↳ PartCrafter K=8 + Particulate | **42.9** | 42.9 | nan | **13.4** | 19.9 | 0.5 | **4.6** | 3.3 | 44.2 | 37.9 / nan / 86.0 | 50 |
| PartPacker | **51.1** | 51.1 | nan | **25.0** | 33.4 | 9.3 | **0.0** | 0.0 | 0.0 | 46.5 / nan / 0.0 | 50 |
| ↳ PartPacker + Particulate | **51.2** | 51.2 | nan | **16.5** | 24.2 | 0.7 | **2.2** | 1.4 | 39.2 | 46.6 / nan / 87.3 | 50 |
| Cube3D → CubePart | **29.7** | 35.3 | 20.4 | **55.7** | 63.8 | 39.0 | **0.0** | 0.0 | 0.0 | 35.4 / 73.3 / 0.0 | 50 |
| ↳ CubePart + Particulate | **47.0** | 47.0 | nan | **16.6** | 24.4 | 0.0 | **6.1** | 4.2 | 47.1 | 44.9 / nan / 89.1 | 50 |
| BrickGPT | **32.3** | 32.3 | nan | **40.7** | 60.9 | 0.0 | **0.0** | 0.0 | 0.0 | 31.4 / nan / 0.0 | 50 |
| LegoACE | **24.9** | 24.9 | nan | **32.5** | 48.8 | 0.0 | **0.0** | 0.0 | 0.0 | 21.9 / nan / 0.0 | 50 |
| PhysX-Anything | **23.7** | 32.8 | 9.1 | **22.8** | 33.7 | 1.3 | **8.3** | 5.7 | 32.4 | 34.5 / 15.2 / 59.5 | 48 |

### Tier A (catalogue parts only)

| system | 2.1 | shape | part geo. | 2.2 | visible | internal | 2.3 | recall | precision | v3.2 2.1 / 2.2 / 2.3 | N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *Closed-source APIs* | | | | | | | | | | | |
| Claude Opus 5 | **67.7** | 75.8 | 55.5 | **53.6** | 74.0 | 10.1 | **50.1** | 45.9 | 76.1 | 77.4 / 49.7 / 88.9 | 20 |
| Claude Fable 5.1 | **79.5** | 84.3 | 72.4 | **62.1** | 85.5 | 12.4 | **55.9** | 51.8 | 77.3 | 86.8 / 54.0 / 88.9 | 20 |
| Claude Sonnet 5 | **56.2** | 61.9 | 47.0 | **60.9** | 81.2 | 19.9 | **44.8** | 38.9 | 68.6 | 63.4 / 50.9 / 94.8 | 50 |
| Claude Haiku 4.5 | **39.6** | 45.5 | 30.4 | **59.2** | 84.6 | 8.1 | **30.0** | 25.1 | 57.7 | 46.6 / 47.5 / 83.7 | 50 |
| *Open-source models* | | | | | | | | | | | |
| Qwen3.5-27B | **44.5** | 51.0 | 34.1 | **52.2** | 73.8 | 8.4 | **23.5** | 18.6 | 56.5 | 53.1 / 43.5 / 84.0 | 50 |
| Qwen3.5-35B-A3B | **38.5** | 48.3 | 27.1 | **48.8** | 73.4 | 5.0 | **14.1** | 11.7 | 41.9 | 48.2 / 40.8 / 71.5 | 50 |
| gpt-oss-120B | **32.4** | 37.3 | 24.9 | **47.6** | 67.3 | 7.3 | **20.6** | 16.0 | 51.8 | 39.1 / 39.0 / 84.2 | 50 |
| Gemma-4-31B-IT | **47.8** | 52.6 | 39.8 | **45.9** | 66.2 | 5.0 | **23.1** | 18.7 | 50.9 | 55.2 / 38.5 / 75.3 | 50 |
| Qwen3-VL-32B | **31.6** | 40.3 | 24.0 | **43.8** | 67.6 | 6.4 | **13.0** | 11.6 | 43.9 | 36.7 / 37.0 / 72.7 | 50 |
| Qwen3-VL-30B-A3B | **28.2** | 35.7 | 15.1 | **44.7** | 64.0 | 5.0 | **3.3** | 2.7 | 12.2 | 37.2 / 36.9 / 32.7 | 50 |
| Qwen3-VL-8B | **27.2** | 34.1 | 17.6 | **33.8** | 49.6 | 3.1 | **4.4** | 3.6 | 26.9 | 34.5 / 29.9 / 67.3 | 50 |
| Qwen3-4B-Instruct | **20.9** | 33.0 | 10.7 | **27.5** | 46.0 | 3.1 | **6.8** | 6.5 | 29.6 | 29.2 / 25.9 / 53.5 | 50 |
| InternVL3.5-38B | **29.9** | 35.3 | 20.1 | **39.6** | 57.0 | 3.4 | **7.2** | 5.8 | 28.1 | 36.8 / 31.2 / 60.8 | 50 |
| InternVL3.5-8B | **22.2** | 28.8 | 14.6 | **33.1** | 50.2 | 3.5 | **1.3** | 0.8 | 22.7 | 31.2 / 28.4 / 48.5 | 50 |
| MiniCPM-V-4.5 | **25.0** | 31.3 | 14.5 | **28.3** | 43.3 | 3.8 | **0.0** | 0.0 | 8.5 | 28.2 / 22.5 / 30.4 | 50 |
| ERNIE-4.5-VL-28B-A3B | **16.5** | 27.6 | 9.6 | **29.3** | 53.2 | 2.2 | **2.8** | 2.9 | 22.5 | 21.2 / 25.2 / 42.8 | 50 |
| Ministral-3-8B | **25.4** | 31.7 | 15.5 | **50.1** | 72.9 | 3.8 | **3.8** | 3.0 | 26.0 | 31.0 / 39.1 / 54.1 | 50 |

### Tier C (created parts only)

| system | 2.1 | shape | part geo. | 2.2 | visible | internal | 2.3 | recall | precision | v3.2 2.1 / 2.2 / 2.3 | N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *Closed-source APIs* | | | | | | | | | | | |
| Claude Opus 5 | **65.6** | 71.7 | 56.3 | **59.3** | 77.3 | 21.1 | **55.4** | 50.7 | 78.1 | 74.2 / 49.9 / 88.8 | 20 |
| Claude Fable 5.1 | **70.1** | 75.1 | 62.4 | **70.0** | 88.1 | 32.1 | **55.0** | 50.1 | 71.9 | 75.6 / 56.2 / 88.4 | 20 |
| Claude Sonnet 5 | **49.7** | 55.5 | 40.3 | **59.2** | 77.7 | 22.6 | **35.0** | 32.5 | 59.0 | 55.7 / 50.6 / 91.7 | 50 |
| Claude Haiku 4.5 | **36.8** | 42.3 | 28.0 | **50.6** | 70.1 | 11.6 | **15.6** | 11.6 | 45.7 | 42.3 / 42.6 / 85.9 | 50 |
| *Open-source models* | | | | | | | | | | | |
| Qwen3.5-27B | **41.2** | 47.1 | 32.0 | **48.2** | 64.8 | 15.0 | **14.4** | 11.3 | 34.7 | 47.4 / 41.4 / 71.1 | 50 |
| Qwen3.5-35B-A3B | **31.0** | 36.4 | 24.1 | **51.3** | 70.0 | 15.8 | **13.6** | 11.1 | 35.3 | 34.7 / 41.9 / 69.5 | 50 |
| gpt-oss-120B | **29.3** | 34.1 | 22.1 | **50.8** | 69.5 | 12.5 | **21.6** | 18.6 | 45.1 | 35.7 / 43.2 / 78.2 | 50 |
| Gemma-4-31B-IT | **38.9** | 46.1 | 27.0 | **45.8** | 63.2 | 11.0 | **16.3** | 12.6 | 38.3 | 46.2 / 40.1 / 83.4 | 50 |
| Qwen3-VL-32B | **15.7** | 33.6 | 14.7 | **28.3** | 65.1 | 10.8 | **7.3** | 9.9 | 27.1 | 17.8 / 25.8 / 38.2 | 50 |
| Qwen3-VL-30B-A3B | **18.4** | 28.9 | 12.2 | **37.0** | 61.5 | 13.2 | **2.2** | 2.0 | 9.3 | 21.0 / 32.3 / 25.2 | 50 |
| Qwen3-VL-8B | **16.2** | 29.7 | 11.2 | **30.0** | 59.8 | 6.3 | **3.7** | 3.6 | 22.5 | 20.0 / 28.2 / 53.3 | 50 |
| Qwen3-4B-Instruct | **20.0** | 28.8 | 8.6 | **35.8** | 51.1 | 11.6 | **2.2** | 2.1 | 19.4 | 23.8 / 33.3 / 65.6 | 50 |
| InternVL3.5-38B | **20.3** | 25.7 | 11.5 | **29.6** | 42.2 | 7.6 | **6.4** | 5.6 | 16.7 | 22.5 / 26.2 / 48.6 | 50 |
| InternVL3.5-8B | **7.8** | 22.6 | 5.7 | **8.7** | 23.9 | 6.9 | **0.0** | 0.0 | 2.4 | 9.7 / 9.4 / 2.1 | 50 |
| MiniCPM-V-4.5 | **13.5** | 23.7 | 6.7 | **22.7** | 43.1 | 4.8 | **0.0** | 0.0 | 8.2 | 16.2 / 18.2 / 13.6 | 50 |
| ERNIE-4.5-VL-28B-A3B | **13.5** | 25.3 | 9.3 | **19.6** | 40.0 | 3.4 | **0.7** | 1.0 | 4.5 | 16.9 / 17.8 / 19.0 | 50 |
| Ministral-3-8B | **21.0** | 26.8 | 11.9 | **43.2** | 57.6 | 13.6 | **3.1** | 2.1 | 17.1 | 23.4 / 40.2 / 40.3 | 50 |

### Separation (Tier B, 19 LLMs)

| | 2.1 | 2.2 | 2.3 | v3.2 2.1 | v3.2 2.2 | v3.2 2.3 |
|---|---|---|---|---|---|---|
| closed mean | 61.0 | 63.1 | 46.1 | 67.1 | 54.4 | 91.1 |
| open mean | 26.5 | 38.8 | 8.9 | 32.2 | 33.4 | 57.1 |
| closed − open | 34.5 | 24.3 | 37.2 | 34.9 | 21.0 | 34.0 |
| sd over the 19 | 18.9 | 14.1 | 19.1 | 19.4 | 12.2 | 23.4 |
| top-5 closed spread (max − min) | 21.0 | 3.7 | 11.2 | 21.8 | 7.7 | 3.0 |

### Closed models on their 40 common Tier B tasks

| system | 2.1 | shape | part geo. | 2.2 | 2.3 |
|---|---|---|---|---|---|
| GPT-6 Astra | 76.7 | 83.4 | 66.7 | 65.2 | 54.8 |
| GPT-5.6 Sol | 62.6 | 68.2 | 54.2 | 64.5 | 52.6 |
| Claude Opus 5 | 65.4 | 71.0 | 56.9 | 65.3 | 48.2 |
| Claude Fable 5.1 | 65.9 | 69.6 | 60.3 | 63.5 | 54.1 |
| Claude Sonnet 5 | 61.8 | 66.0 | 55.6 | 61.7 | 45.7 |
| Claude Haiku 4.5 | 41.2 | 47.3 | 32.1 | 58.5 | 25.4 |

### Created geometry (reported, not scored): share of the design's surface that is created, and its precision

| system | Tier B created share | Tier B created precision | Tier C shape | Tier B shape |
|---|---|---|---|---|
| GPT-6 Astra | 9.4 | 73.7 | nan | 83.4 |
| GPT-5.6 Sol | 25.0 | 65.0 | nan | 68.2 |
| Claude Opus 5 | 53.6 | 70.7 | 71.7 | 71.0 |
| Claude Fable 5.1 | 31.6 | 60.5 | 75.1 | 69.6 |
| Claude Sonnet 5 | 22.7 | 47.6 | 55.5 | 61.2 |
| Claude Haiku 4.5 | 13.0 | 57.0 | 42.3 | 45.9 |
| Qwen3.5-27B | 66.4 | 51.6 | 47.1 | 52.8 |
| Qwen3.5-35B-A3B | 22.4 | 40.2 | 36.4 | 45.3 |
| gpt-oss-120B | 26.9 | 40.4 | 34.1 | 37.0 |
| Gemma-4-31B-IT | 49.0 | 36.1 | 46.1 | 47.1 |
| Qwen3-VL-32B | 59.7 | 37.9 | 33.6 | 38.3 |
| Qwen3-VL-30B-A3B | 43.1 | 27.6 | 28.9 | 32.0 |
| Qwen3-VL-8B | 62.8 | 22.7 | 29.7 | 26.6 |
| Qwen3-4B-Instruct | 80.1 | 27.2 | 28.8 | 29.8 |
| InternVL3.5-38B | 26.8 | 36.7 | 25.7 | 32.0 |
| InternVL3.5-8B | 36.9 | 28.8 | 22.6 | 27.8 |
| MiniCPM-V-4.5 | 21.4 | 33.0 | 23.7 | 26.8 |
| ERNIE-4.5-VL-28B-A3B | 8.1 | 21.2 | 25.3 | 27.7 |
| Ministral-3-8B | 22.8 | 38.1 | 26.8 | 28.7 |

## Reading the numbers

- **2.1.**
  - GPT-6 Astra leads Tier B clearly: 76.7, against 65.9 for Fable and 65.4 for Opus on the same 40 tasks. It has
    the best shape (83.4) and the best part geometry (66.7).
  - Removing the creation term took away the penalty for GPT-6's 9% created share. What it does create is the most
    precise of any system (73.7).
  - On Tier C, where every part is created, Fable's shape (75.1) is above its own Tier B shape (69.6). Creation is
    not penalised.
- **2.2.**
  - Visible parts sit at 83–87 for the top five, and internal parts at 18–26.
  - The top five are within 3.7 points of each other, and 2.2 is where they are closest.
  - Geometric placement grounds about 10% of LLM parts that their names missed.
  - Unnamed interior modules are rare in agent designs: 27 of 850 Tier B designs. The agents seldom model hidden
    parts at all.
- **2.3.**
  - Closed mean 46 against open 9. The top five span 11 points.
  - GPT-6 and Fable lead Tier B (54.8 and 54.1).
  - Recall is 40–52 for the top five, and precision 66–77.
  - 47% of Tier B joints realise a target or claim. For the baselines it is 14%.
  - Particulate adds joints that sound mechanically but rarely sit on a target part, so it scores 2–6.

## Known limitations

1. **Lexicon gaps are left unmatched on purpose** (no synonym table was added):
   - `upper arm` / `forearm` against the robot arm's `links`;
   - a trackball's ball named `trackball`;
   - `front frame` / `rear frame` against `ladder halves`.

   Geometric placement recovers such parts for 2.2 only when the reference labels them.
2. **Tasks with no motion target** are skipped in 2.3: dining chair, bed frame, dining table and stepladder. The first
   three have no claimed motion and no annotated joint. The stepladder's `ladder halves` is not grounded.
3. **bc_tower_crane**: every system's shape is below 20. The LEGO reference's long jib and lattice set proportions
   that no design matches.
4. **Different evidence for different families.** LLM designs are grounded by name first. Mesh generators are
   grounded by placement only, so they can never show an internal part except as an unnamed interior module.
5. **GPT-6 Astra and GPT-5.6 Sol have no Tier A or Tier C runs.** Opus and Fable have 20 tasks on those tiers.

## Full table, all systems and tiers (spec v3.7, x100)

`0` without a decimal means no run exists for that system and tier. `0.0` is a real score of zero. N is the number of tasks. Domain baselines have no tier, so they appear only in the Base columns.

| system | A 2.1 | A 2.2 | A 2.3 | A N | B 2.1 | B 2.2 | B 2.3 | B N | C 2.1 | C 2.2 | C 2.3 | C N | Base 2.1 | Base 2.2 | Base 2.3 | Base N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Closed-source APIs* | | | | | | | | | | | | | | | | |
| GPT-6 Astra | 0 | 0 | 0 | 0 | 76.7 | 65.2 | 54.8 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GPT-5.6 Sol | 0 | 0 | 0 | 0 | 62.6 | 64.5 | 52.6 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Claude Opus 5 | 67.7 | 53.6 | 50.1 | 20 | 65.4 | 65.3 | 48.2 | 40 | 65.6 | 59.3 | 55.4 | 20 | 0 | 0 | 0 | 0 |
| Claude Fable 5.1 | 79.5 | 62.1 | 55.9 | 20 | 65.9 | 63.5 | 54.1 | 40 | 70.1 | 70.0 | 55.0 | 20 | 0 | 0 | 0 | 0 |
| Claude Sonnet 5 | 56.2 | 60.9 | 44.8 | 50 | 55.7 | 61.6 | 43.6 | 50 | 49.7 | 59.2 | 35.0 | 50 | 0 | 0 | 0 | 0 |
| Claude Haiku 4.5 | 39.6 | 59.2 | 30.0 | 50 | 39.4 | 58.4 | 23.3 | 50 | 36.8 | 50.6 | 15.6 | 50 | 0 | 0 | 0 | 0 |
| *Open-source models* | | | | | | | | | | | | | | | | |
| Qwen3.5-27B | 44.5 | 52.2 | 23.5 | 50 | 46.3 | 53.0 | 20.8 | 50 | 41.2 | 48.2 | 14.4 | 50 | 0 | 0 | 0 | 0 |
| Qwen3.5-35B-A3B | 38.5 | 48.8 | 14.1 | 50 | 36.9 | 50.3 | 17.6 | 50 | 31.0 | 51.3 | 13.6 | 50 | 0 | 0 | 0 | 0 |
| gpt-oss-120B | 32.4 | 47.6 | 20.6 | 50 | 31.2 | 50.2 | 16.7 | 50 | 29.3 | 50.8 | 21.6 | 50 | 0 | 0 | 0 | 0 |
| Gemma-4-31B-IT | 47.8 | 45.9 | 23.1 | 50 | 41.1 | 46.9 | 17.3 | 50 | 38.9 | 45.8 | 16.3 | 50 | 0 | 0 | 0 | 0 |
| Qwen3-VL-32B | 31.6 | 43.8 | 13.0 | 50 | 23.3 | 36.8 | 7.4 | 50 | 15.7 | 28.3 | 7.3 | 50 | 0 | 0 | 0 | 0 |
| Qwen3-VL-30B-A3B | 28.2 | 44.7 | 3.3 | 50 | 24.5 | 41.7 | 3.6 | 50 | 18.4 | 37.0 | 2.2 | 50 | 0 | 0 | 0 | 0 |
| Qwen3-VL-8B | 27.2 | 33.8 | 4.4 | 50 | 16.4 | 30.8 | 4.8 | 50 | 16.2 | 30.0 | 3.7 | 50 | 0 | 0 | 0 | 0 |
| Qwen3-4B-Instruct | 20.9 | 27.5 | 6.8 | 50 | 23.4 | 38.2 | 3.9 | 50 | 20.0 | 35.8 | 2.2 | 50 | 0 | 0 | 0 | 0 |
| InternVL3.5-38B | 29.9 | 39.6 | 7.2 | 50 | 24.1 | 33.4 | 9.9 | 50 | 20.3 | 29.6 | 6.4 | 50 | 0 | 0 | 0 | 0 |
| InternVL3.5-8B | 22.2 | 33.1 | 1.3 | 50 | 15.5 | 19.4 | 2.0 | 50 | 7.8 | 8.7 | 0.0 | 50 | 0 | 0 | 0 | 0 |
| MiniCPM-V-4.5 | 25.0 | 28.3 | 0.0 | 50 | 20.6 | 28.0 | 2.3 | 50 | 13.5 | 22.7 | 0.0 | 50 | 0 | 0 | 0 | 0 |
| ERNIE-4.5-VL-28B-A3B | 16.5 | 29.3 | 2.8 | 50 | 17.4 | 28.7 | 4.0 | 50 | 13.5 | 19.6 | 0.7 | 50 | 0 | 0 | 0 | 0 |
| Ministral-3-8B | 25.4 | 50.1 | 3.8 | 50 | 23.2 | 46.4 | 5.5 | 50 | 21.0 | 43.2 | 3.1 | 50 | 0 | 0 | 0 | 0 |
| *Domain-specific baselines* | | | | | | | | | | | | | | | | |
| PartCrafter (K=15) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45.8 | 27.3 | 0.0 | 50 |
| ↳ PartCrafter K=15 + Particulate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45.8 | 11.0 | 2.5 | 50 |
| PartCrafter (K=8) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 41.3 | 20.8 | 0.0 | 50 |
| ↳ PartCrafter K=8 + Particulate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 42.9 | 13.4 | 4.6 | 50 |
| PartPacker | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 51.1 | 25.0 | 0.0 | 50 |
| ↳ PartPacker + Particulate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 51.2 | 16.5 | 2.2 | 50 |
| Cube3D → CubePart | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 29.7 | 55.7 | 0.0 | 50 |
| ↳ CubePart + Particulate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 47.0 | 16.6 | 6.1 | 50 |
| BrickGPT | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32.3 | 40.7 | 0.0 | 50 |
| LegoACE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24.9 | 32.5 | 0.0 | 50 |
| PhysX-Anything | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 23.7 | 22.8 | 8.3 | 48 |
