# LMBuild demos

Open `index.html` for everything with thumbnails. The pages work offline. The only external requests are Google Fonts, which fall back to system fonts.

Unless a caption says otherwise, a demo uses the **main setting**: condition `name_only+image`, one 3-round trajectory (`r3`), seed 0.
Model names are the run keys (`<system>__<tier>__<condition>__r3__s0`). Episodes come from `results/v2/<task>/runs/<namespace>/`. The namespaces are `core_v2.4` (main), `core_v2.4_clean` (GPT Tier B) and `core_v2.3` (older open-model Tier B).

## 1. `trajectories/`: 13 episodes

`trajectories/index.html` lists all 13. Each `<case>/index.html` is the built page. `<case>/raw_episode/` holds the stored record of the episode it replays:
`trace.json` (every tool call and its args), `state.json`, `session_state.json`, `claude_run.json` (Claude arms only: cost, turns, session id), `design/` (final `design.json` + `design.glb`) and `rounds/round_<n>/` (per-round checkpoint: design, state, trace, submit result). `SOURCE_RUN_DIR.txt` names the original run dir.
None of these 13 are open-model runs, so none has a `transcript.json` or `loop_log.json`. The pages' transcript extracts come from the agents' CLI logs (Claude Code session / Codex rollout). Those private logs are not copied.

Full-trajectory replays: every tool call, the observation the agent read, a render after each design change, round reviews, final design, joints, assembly sequence and part provenance. `agent_trajectories.html` is their index. The first five were picked by the v3.1 ranking. The wheelchair was requested as a walkthrough.

| case | task | model | tier | namespace |
|---|---|---|---|---|
| `washing_machine__gpt-6-astra__B` | washing_machine | gpt-6-astra | B | core_v2.4_clean |
| `kitchen_oven__claude-opus-5__B` | kitchen_oven | claude-opus-5 | B | core_v2.4 |
| `camera__gpt-6-astra__B` | camera | gpt-6-astra | B | core_v2.4_clean |
| `camera__claude-sonnet-5__C` | camera | claude-sonnet-5 | C | core_v2.4 |
| `oscillating_fan__claude-sonnet-5__C` | oscillating_fan | claude-sonnet-5 | C | core_v2.4 |
| `wheelchair__gpt-6-astra__B` | wheelchair | gpt-6-astra | B | core_v2.4_clean (white style) |

`create_part` showcases: one created part each, with the model's verbatim command and CSG program, before/after each placement, the final design, the reference object and the condition image. `part_creation_examples.html` is their index.

| case | task | model | tier | part |
|---|---|---|---|---|
| `steam_engine_flywheel__claude-opus-5__C` | oscillating_steam_engine | claude-opus-5 | C | flywheel |
| `tractor_front_tire__claude-fable-5.1__C` | farm_tractor | claude-fable-5.1 | C | front_tire |
| `desk_lamp_shade__claude-opus-5__B` | desk_lamp | claude-opus-5 | B | lamp_shade_dome (2 failed attempts, then fixed) |
| `car_jack_saddle__claude-sonnet-5__C` | scissor_car_jack | claude-sonnet-5 | C | top_saddle |
| `door_panel__claude-haiku-4.5__C` | hinged_door | claude-haiku-4.5 | C | door_panel_6pane |
| `fan_rotor__gpt-5.6-sol__B` | oscillating_fan | gpt-5.6-sol | B | three_blade_rotor |
| `cart_caster_fork__gpt-6-astra__B` | shopping_cart | gpt-6-astra | B | caster_fork |

## 2. `outcomes/`: final-design galleries

- `bright10/index.html`: 10 objects, the **round-1** design of every agent in Tier B and C plus the domain-specific generators. Iso view in role colours (WebP sprites) and 3-pose joint strips.
- `demo_four_objects.html`: office_desk, toilet, bookcase and wheelchair, round 1, all models, Tier B vs C vs generators.
- `demo_objects.html`: the candidate easy objects, iso and front views.
- `<task>_round1/`: round-1 renders per model. `B/`, `C/` and `EXT/` hold `<system>_{iso,front,side}.jpg`. `sheet_*.jpg` are contact sheets. Some of these folders also have an `index.html`.
- `joints_white/`: 35 three-pose joint strips (lower / middle / upper limit, fixed camera), round-1 designs.
- `realistic/materials_10.html` and `realistic/wheelchair_materials.html`: material renders. Each declared submaterial maps to a PBR preset, with one default colour per material.
- `final_designs/<task>/<closed|open>__<system>/`: `design.glb`, `design.json` and `SOURCE.json` for the best closed and the best open model per showcase object. All are Tier B, main setting, final submitted design (the round-3 checkpoint).
  Each pick has the highest **round-3** `headline.overall` (eval spec v3.4.5, `results/v2/<task>/eval_v345/<key>__round3.json`) with no critical fail in its class. Closed = claude-\* and gpt-5.6-sol / gpt-6-astra. Open = open-weight models, with GLM and Kimi excluded.

| task | closed (round-1 / round-3 overall) | open (round-1 / round-3 overall) |
|---|---|---|
| office_desk | gpt-5.6-sol, core_v2.4_clean (0.916 / 0.943) | gemma-4-31b-it, core_v2.4 (0.747 / 0.914) |
| toilet | gpt-5.6-sol, core_v2.4_clean (0.934 / 0.934) | qwen3.5-27b, core_v2.4 (0.885 / 0.879) |
| bookcase | gpt-6-astra, core_v2.4_clean (0.936 / 0.972) | qwen3-vl-32b-instruct, core_v2.4 (0.883 / 0.882) |
| wheelchair | gpt-6-astra, core_v2.4_clean (0.883 / 0.880) | qwen3.5-27b, core_v2.4 (0.768 / 0.807) |

## 3. `gifs/`: animations (looping, 560 px wide, under 1.5 MB each)

- `assembly/`: nine assembly GIFs, built from the existing step pictures.
  - Six full trajectories: one frame per design-changing tool call (`frames/step_NNN_iso.jpg`), captioned with step, round, tool and instance.
  - Three `create_part` showcases (tractor tyre, cart caster fork, fan rotor): the part, then before/after each placement, then the final design.
  - Metadata: `assembly_gifs.json`.
- `joints/`: 14 joint-motion GIFs, **re-rendered** for this release. There are 9 poses eased from the lower limit to the upper limit, played back to the lower limit (16 frames). The camera is fixed, framed on the union of all poses. The renderer and colours are the same as `outcomes/joints_white` (white style, CPU Cycles). Moving parts are outlined and the orange rod is the declared axis. The moving set comes from the evaluator's `kinematic_moving_set`. Agent joints come from **round-1** designs. One joint (`toilet__physx-anything__ext__g2`) comes from the PhysX-Anything generator. A continuous joint with no limits (excavator slew) sweeps 0°→120°.
- `turntable/`: four turntables, re-rendered at 18 azimuths with 24° elevation. They show the final Tier B designs of office_desk (gpt-5.6-sol), bookcase (gpt-6-astra), wheelchair (gpt-6-astra) and toilet (qwen3.5-27b), the same runs as `outcomes/final_designs`.
- Metadata for `joints/` and `turntable/`: `render_gifs.json`.

## Notes

- The five grey-style trajectory pages (all except the wheelchair) show kitchen_oven catalog parts P013 and P005 in solid black. The catalog meshes store every face twice, and the harness renders show them the same way. Only the white-style renders dedupe the faces for display.
- The trajectory pages quote the agents' prompts and commands verbatim, except that the build machine's repository root was stripped from absolute paths (so they read as repo-relative paths, e.g. `results/v2/...`, and some point to the pre-release layout `reference_data/...`). They are text, not links. Every image the pages show is a local copy.
- Byte-identical large files inside `trajectories/` are hard links, for example `rounds/round_3/design/design.glb` and `design/design.glb`. Design GLBs over 40 MB (the kitchen_oven Opus episode, ~870 MB each, and the fan-rotor GPT-5.6 episode, ~230 MB each, which embed high-poly catalogue meshes) are omitted; a `design.glb.OMITTED.txt` stands in their place, and `design.json` + `trace.json` rebuild them.
