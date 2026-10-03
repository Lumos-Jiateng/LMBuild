# Environment and tools

An episode asks a system to build one task's object as an assembly of posed part meshes, with declared roles,
materials, joints and an assembly sequence. The environment is `ppbench/v2/env.py` (`AssemblyEnv`, protocol v2.2); the
exact text every agent receives is in `docs/environment/prompt_office_desk_tier{A,B,C}.txt`.

## World and design

- Metres, +Z up, ground plane z = 0, the object's front faces −Y.
- A part is a mesh with a pose. Catalogue parts come from the task's subpart pool
  (`benchmark/tasks/<task>/pool/`), each centred on its bounding box. Created parts are compiled from a JSON CSG program
  (`ppbench/v2/csg.py`, manifold3d).
- Parts are connected when their real surfaces touch (gap < ~4 mm) or a joint joins them where both are (origin within
  2 cm of each part's surface).
- A design (`ppbench/v2/design.py`) is saved as `design.json` (parts with role, material, source, pose; joints; sequence)
  plus `design.glb` (the geometry).

## Tiers, conditions, rounds

| | catalogue (pool) | may create parts |
|---|---|---|
| Tier A | yes | no |
| Tier B | yes | yes (`create_part`) |
| Tier C | no | yes, every part |

Six conditions per task: `name_only`, `attributes`, `functional`, each with or without the in-the-wild image
(`<prompt>+image`: "According to image W1, build …"). An episode has 1–3 rounds; `submit` ends a round and, if rounds
remain, returns a review (the `check` report plus renders) before the agent revises the same design. The main setting of
the paper is Tier B, `name_only+image`, one 3-round trajectory, seed 0.

## Tools

| group | tool | notes |
|---|---|---|
| catalogue | `list_parts(query?)`, `inspect_part(part_id)`, `view_catalog_sheet()`, `list_materials()` | the sheet is the labelled picture of every pool part |
| geometry | `place_part(part_id, instance_id, position, rotation?, role?, material?)`, `move_part`, `remove_part`, `attach(instance, anchor, target, target_anchor, offset?)`, `measure(a, b)` | `measure` uses real surfaces, not boxes |
| creation | `create_part(name, program)` | Tier B and C; box / cylinder / sphere / torus / extrude / revolve, union / difference / intersection / hull / transform / mirror |
| declarations | `set_part_info(instance, role?, material?)`, `add_joint(joint_id, type, parent, child, axis, origin, limits?)`, `remove_joint`, `set_assembly_sequence(steps)` | joint types: fixed, revolute, continuous, prismatic, cylindrical, ball |
| self-check | `get_scene()`, `check()` (metered), `render(views?)` (metered) | views: cond, iso, front, side, back_iso, top |
| flow | `submit()` | ends a round |

Tool calls, `check` and `render` are budgeted; `get_scene` reports the budget left (limits in `ppbench/v2/env.py`).

## Two transports over the same environment

- **Shell session** (`ppbench/v2/session.py`), for terminal agents. State lives in one JSON file; every call reloads it,
  dispatches through `AssemblyEnv` and saves:

      .venv_eval/bin/python -m ppbench.v2.session init   --state S.json --task office_desk --tier B \
          --condition name_only+image --rounds 3 --out <run_dir> --system <name> \
          --snapshot results/v2/office_desk/task_snapshot_core.json
      .venv_eval/bin/python -m ppbench.v2.session prompt --state S.json          # the agent's full instructions
      .venv_eval/bin/python -m ppbench.v2.session call   --state S.json <tool> '<JSON args>'
      .venv_eval/bin/python -m ppbench.v2.session finish --state S.json          # export what was built

  The Claude arms ran through it with headless Claude Code (`ppbench/v2/claude_batch.py`, whose only allowed tools are
  this command and reading image files), the GPT arms with the Codex CLI (`scripts/run_codex_episode.py`). Each episode
  ran from its own working directory outside the repository.
- **OpenAI-compatible loop** (`ppbench/v2/loop.py`), for served open models: text tool calls parsed from the reply
  (`toolcalls.py`, with repair of truncated JSON), images attached to the user turn, up to 3 nudges when a reply has no
  tool call. `ppbench/v2/core_batch.py` runs a model over the task set; `scripts/run_open_model.sh` serves one of the
  paper's open models from `configs/open_models.tsv` and runs it.

## What an episode stores (`<run_dir>/`)

`trace.json` (every tool call with arguments, ok/error, round), `state.json`, `session_state.json` (shell transport),
`transcript.json` + `loop_log.json` (open models), `claude_run.json` (Claude arms: cost, turns, session id),
`design/design.{json,glb}` and `rounds/round_<n>/` (an immutable checkpoint after every submit), `renders/`.

## Domain-specific generators

`ppbench/baselines/run_*.py` wrap PartCrafter, PartPacker, Cube3D + CubePart, Particulate, PhysX-Anything, BrickGPT,
LegoACE and BrickNet; `ppbench/v2/core_externals.py` runs them over the core set into
`results/v2/<task>/external_core/`, and `ppbench/v2/external.py` turns their outputs into designs (they declare only what
their producer emits; undeclared dimensions are skipped or scored 0 as the metric states). Each needs its upstream
repository and weights installed under `third_party/` (not shipped); the wrappers document the expected layout.

## Known issues (kept as they were during the study)

- `session.py` has no file lock: two `session call`s on one state file in parallel can lose one call's update. Agents
  that issue parallel shell commands (Codex) hit this rarely; the showcase pages flag such calls.
- `create_part` extrude/revolve with a clockwise polygon fails with a "(near) zero area" error (manifold3d gives a CW
  polygon zero area); the prompt does not say that polygons must be counter-clockwise.
- Two kitchen_oven catalogue meshes (P005, P013) store every face twice with both windings and render black in the
  review renders.
- The 10 tasks added last (5 curated CAD products, 5 `bc_*` LEGO models) have no catalogue contact sheet; agents on
  them were not shown one.
