# Human evaluation — Level 3 (3.1 decomposition, 3.2 aesthetics, 3.3 structure alignment)

v2 (2026-09-21). Each item asks one question about one dimension and takes a single 1–5 rating.
One click saves the rating and opens the next item.

| dim | what the annotator sees | question |
|---|---|---|
| 3.2 aesthetics | object name + design sheet (no photo) | Is the design an elegant object of that kind, in line with human aesthetic sense? |
| 3.3 structure alignment | condition photo + design sheet | How faithfully does it reproduce the photographed object's structure? |
| 3.1 decomposition | condition photo + design sheet + part count | Is it broken into parts the way the real product is made and assembled? |

The anchors for 1, 3 and 5 are in `build.py:QUESTIONS`. The photo-or-not choice per dimension follows the VLM judge
(`ppbench/v2/judge_l3.py`). Every dimension uses one design sheet, rendered on white with bright colours in the
`delivery/agent_trajectories --style white` look and one colour per part:
- Top row: the assembled design, front-right and rear-left.
- Bottom row: the exploded design. Designs with more than 40 parts show front and top views instead.

## The set
- **Designs:** Tier B, `name_only+image`, the **round-1 checkpoint** of the 3-round trajectory, seed 0. The two
  domain-specific generators have a single final output.
- **Systems (6):**
  - GPT-6-astra (closed API)
  - Claude Sonnet 5 (closed API)
  - Qwen3.5-27B (open, strong)
  - Gemma-4-31B-it (open, weaker; a different model family)
  - PartPacker (domain-specific, image → part-level 3D)
  - BrickGPT (domain-specific, LEGO)
- **Tasks (15):** offroad_jeep, passenger_car, wheeled_excavator, scissor_car_jack, bench_drill_press,
  oscillating_steam_engine, dining_table, bed_frame, chest_of_drawers, swivel_office_chair, stepladder,
  refrigerator, desk_lamp, oscillating_fan, wheelchair. All 90 designs have at least 3 parts.
  `build.py` refuses any design with fewer than 2.
- **Split:** 90 designs × 3 dims = 270 unique items. Each of the 4 annotators gets **90** (30 per dim).
  - Primary raters follow a Latin square: annotator = (task + system + dim offset) mod 4. Each annotator sees
    every system 12–17 times, and the three dims of one design go to three different people.
  - The 90 spare slots double-annotate 30 items per dimension (5 per system) for inter-annotator agreement.
  - Block order: 3.2 → 3.3 → 3.1, so aesthetics is rated before any photo is shown. Items are shuffled within a block.

## Files
- `build.py`: selects the set, renders `renders/<task>/<key>.png` (cached; `--force` redraws them), copies the
  sheets to `static/img/<opaque id>.jpg`, and writes `static/set.json` and `key.json`.
- `server.py`: stdlib only. Every save is appended to `annotations/<A>.jsonl`, and the latest rating per item
  is kept in `annotations/<A>.json`.
- `key.json`: the unblinding key. `annotators.json`: the link tokens. Neither is served.

## Run
    python3 human_evaluation/server.py --port 8790     # prints the four annotator links and the admin link
- Annotator link: `http://<host>:8790/?t=<token>`
- Progress: `/admin?k=<admin>`. The CSV at `/export.csv?k=<admin>` has system, dim, `score_1_5`,
  `score_norm = (s-1)/4` and time taken.
