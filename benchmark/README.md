# LMBuild core benchmark (200 tasks)

Self-contained copy of `reference_data/benchmark_v2/core_v2.json` and every file it references. All paths inside the JSON files are relative to the release root (the directory that contains `benchmark/`), e.g. `benchmark/tasks/offroad_jeep/images/wild_01.png`. Apart from path strings, `core_v2.json` is identical to the source file (same 200 tasks, same order, same content).

## Layout

```
benchmark/
  core_v2.json             all 200 tasks, canonical order, paths rewritten
  task_ids.txt             200 task ids, one per line, canonical order
  tasks/<task_id>/
    task.json              this task's element of core_v2.json (identical dict)
    images/                in-the-wild condition images (wild_01.png ...)
    reference_renders/     clean reference renders; also the image-generator input images
    reference/             reference.glb + every reference.cad_files entry (LDR, STEP, GLB, JS preview)
      alternates/          LDR files of reference.alternates (LDraw-referenced tasks only)
      joint_renders/       joint visual_check images (umbrella only)
    pool/<pool_part_id>.glb  non-LDraw subpart-pool meshes
    pool_sheet/            pool contact sheets shown to agents (pool_sheet_all.png, pool_sheet_N.png, thumbs/)
    knowledge/             claims_verified.json, claims_extracted.json, prompts.json, sources.json (source bundle)
  assets/
    ldraw/glb/<part>.glb   shared LDraw pool meshes (LEGO pool parts point here)
    ldraw/inset/<stem>.ply collision meshes for every LDraw stem used by the LDR references and LEGO pools
    ldraw/part_aliases.json.xz, ldraw/part_names.json
    artiverse/<cat>/<src>/<mid>/  <mid>.segmented.glb, <mid>.articulations.json, material.json
  MANIFEST.json            per-task file list with sha256 + bytes, totals, missing / skipped / unresolved lists
```

Notes:

- `pool_sheet/` exists only for the 40 tasks of the first release: it is absent for the 10 tasks added on 2026-09-17 (5 curated CAD tasks and 5 `bc_*` tasks) and for the 150 tasks added on 2026-09-27/29. No contact sheet was ever rendered for them, so agents on those tasks were not shown one.
- Pool parts taken from the shared standard-part set (`pools/_standard/S*.glb`) are copied into each task's `pool/` under their `pool_part_id`.
- For the 5 curated tasks of the first release the generator input image (`renders_curated/`) differs from the clean render of the same name; the generator input keeps the plain name and the clean render is stored as `reference_renders/<stem>__renders.png`.
- `reference/joint_renders/`: when several joints of one task render to the same file name (pc_heat_press), the file is stored as `<render-dir>_<name>`.
- Artiverse tasks have no `cad_files`; their annotations are in `assets/artiverse/` (see `MANIFEST.json` -> `artiverse_dirs`). The Artiverse data is gated.
- Not included: BrickForge `runs/blockfeat/lib_2048` (port frames, used only by `PartLib.ports()` as supervision).
- LDraw stems without a collision mesh: 4079b (bc_bulldozer:brickcomposer-261458.ldr).

## Tasks

K = required kinematics (+ optional). Pool = subpart-pool part count. Licence strings are copied verbatim from `reference.license`.

| # | task_id | reference dataset | licence | wild imgs | renders | pool parts | P | A | F | K |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | offroad_jeep | BrickNet corpus (LDraw Official Model Repository derived) | LDraw OMR models, CCAL 2.0 / CC BY 2.0 per the OMR terms | 3 | 4 | 120 | 12 | 6 | 4 | 5 (+0) |
| 2 | passenger_car | BrickNet corpus (LDraw Official Model Repository derived) | LDraw OMR models, CCAL 2.0 / CC BY 2.0 per the OMR terms | 3 | 4 | 134 | 20 | 11 | 5 | 12 (+1) |
| 3 | farm_tractor | BrickNet corpus (LDraw Official Model Repository derived) | LDraw OMR models, CCAL 2.0 / CC BY 2.0 per the OMR terms | 3 | 4 | 95 | 12 | 5 | 3 | 7 (+0) |
| 4 | helicopter | BrickNet corpus (LDraw Official Model Repository derived) | LDraw OMR models, CCAL 2.0 / CC BY 2.0 per the OMR terms | 3 | 4 | 160 | 12 | 6 | 4 | 10 (+0) |
| 5 | wheeled_excavator | BrickNet corpus (LDraw Official Model Repository derived) | LDraw OMR models, CCAL 2.0 / CC BY 2.0 per the OMR terms | 3 | 4 | 97 | 13 | 5 | 3 | 4 (+1) |
| 6 | scissor_car_jack | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 44 | 4 | 4 | 2 | 3 (+0) |
| 7 | oscillating_steam_engine | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 37 | 9 | 6 | 2 | 4 (+0) |
| 8 | bench_drill_press | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 87 | 10 | 6 | 4 | 3 (+3) |
| 9 | dining_chair | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 20 | 3 | 6 | 2 | 0 (+5) |
| 10 | dining_table | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 17 | 5 | 6 | 2 | 1 (+4) |
| 11 | bookcase | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 27 | 2 | 4 | 0 | 1 (+2) |
| 12 | bed_frame | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 20 | 6 | 6 | 2 | 0 (+6) |
| 13 | stepladder | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 14 | 5 | 4 | 2 | 1 (+0) |
| 14 | chest_of_drawers | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 8 | 6 | 3 | 1 (+0) |
| 15 | office_desk | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 24 | 5 | 5 | 2 | 3 (+0) |
| 16 | swivel_office_chair | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 57 | 8 | 4 | 3 | 6 (+1) |
| 17 | hinged_door | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 5 | 5 | 3 | 2 (+4) |
| 18 | casement_window | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 6 | 6 | 4 | 1 (+6) |
| 19 | refrigerator | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 27 | 10 | 6 | 4 | 3 (+0) |
| 20 | microwave_oven | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 8 | 8 | 4 | 3 (+0) |
| 21 | kitchen_oven | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 24 | 7 | 6 | 2 | 2 (+1) |
| 22 | washing_machine | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 47 | 7 | 5 | 3 | 6 (+0) |
| 23 | dishwasher | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 9 | 4 | 2 | 3 (+0) |
| 24 | oscillating_fan | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 27 | 9 | 5 | 3 | 2 (+0) |
| 25 | desk_lamp | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 9 | 4 | 3 | 4 (+0) |
| 26 | toilet | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 9 | 6 | 3 | 2 (+0) |
| 27 | mixer_faucet | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 7 | 7 | 2 | 4 (+2) |
| 28 | tool_box | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 5 | 5 | 2 | 4 (+0) |
| 29 | wheeled_dumpster | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 4 | 5 | 2 | 2 (+0) |
| 30 | skateboard | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 8 | 8 | 3 | 2 (+0) |
| 31 | baby_stroller | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 34 | 6 | 5 | 2 | 3 (+0) |
| 32 | shopping_cart | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 8 | 6 | 2 | 2 (+0) |
| 33 | wheelchair | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 34 | 8 | 6 | 4 | 6 (+1) |
| 34 | rolling_suitcase | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 27 | 5 | 6 | 2 | 3 (+0) |
| 35 | umbrella | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 40 | 5 | 8 | 1 | 2 (+1) |
| 36 | scissors | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 3 | 5 | 1 | 2 (+0) |
| 37 | laptop | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 15 | 12 | 6 | 5 | 3 (+0) |
| 38 | camera | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 8 | 5 | 3 | 3 (+2) |
| 39 | monitor_arm | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 2 | 6 | 0 | 5 (+0) |
| 40 | barbecue_grill | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 24 | 3 | 4 | 4 | 3 (+0) |
| 41 | spinal_quadruped_robot | Curated real-world CAD sources | (none given) | 1 | 5 | 278 | 8 | 6 | 3 | 3 (+0) |
| 42 | robot_arm | Curated real-world CAD sources | (none given) | 1 | 5 | 90 | 7 | 5 | 4 | 4 (+0) |
| 43 | computer_mouse | Curated real-world CAD sources | (none given) | 1 | 5 | 26 | 5 | 2 | 1 | 3 (+2) |
| 44 | trackball | Curated real-world CAD sources | (none given) | 1 | 5 | 26 | 4 | 4 | 2 | 2 (+0) |
| 45 | headphones | Curated real-world CAD sources | (none given) | 1 | 5 | 82 | 9 | 6 | 3 | 2 (+0) |
| 46 | bc_tower_crane | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 1 | 4 | 100 | 10 | 6 | 4 | 3 (+0) |
| 47 | bc_fire_engine | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 1 | 4 | 152 | 13 | 5 | 4 | 7 (+1) |
| 48 | bc_forklift | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 1 | 4 | 105 | 12 | 8 | 4 | 5 (+0) |
| 49 | bc_wind_turbine | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 1 | 4 | 120 | 10 | 8 | 5 | 5 (+0) |
| 50 | bc_bulldozer | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 1 | 4 | 204 | 8 | 5 | 3 | 5 (+1) |
| 51 | f360_arbor_press | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 35 | 6 | 6 | 1 | 4 (+0) |
| 52 | f360_toggle_clamp | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 18 | 0 | 3 | 0 | 1 (+0) |
| 53 | f360_bench_vise | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 35 | 6 | 5 | 2 | 4 (+0) |
| 54 | f360_lathe | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 56 | 6 | 4 | 4 | 7 (+0) |
| 55 | f360_concrete_mixer | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 15 | 4 | 4 | 1 | 2 (+0) |
| 56 | f360_microscope | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 26 | 10 | 6 | 5 | 4 (+0) |
| 57 | f360_telescope | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 68 | 5 | 5 | 3 | 3 (+0) |
| 58 | f360_binoculars | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 45 | 6 | 4 | 1 | 4 (+0) |
| 59 | f360_quadcopter | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 40 | 6 | 5 | 2 | 2 (+0) |
| 60 | f360_espresso_machine | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 78 | 9 | 6 | 2 | 2 (+0) |
| 61 | f360_clock | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 146 | 6 | 4 | 2 | 4 (+2) |
| 62 | f360_robotic_gripper | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 50 | 4 | 4 | 1 | 0 (+0) |
| 63 | f360_gear_pump | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 33 | 5 | 6 | 2 | 1 (+0) |
| 64 | f360_newtons_cradle | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 28 | 1 | 5 | 1 | 2 (+0) |
| 65 | f360_globe_valve | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 28 | 7 | 8 | 2 | 3 (+0) |
| 66 | f360_beam_engine | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 135 | 7 | 4 | 1 | 6 (+0) |
| 67 | f360_hand_drill | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 50 | 4 | 5 | 2 | 5 (+0) |
| 68 | f360_pumpjack | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 228 | 14 | 5 | 3 | 8 (+0) |
| 69 | f360_air_compressor | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 248 | 9 | 7 | 3 | 8 (+0) |
| 70 | f360_scissor_lift | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 133 | 8 | 5 | 4 | 5 (+0) |
| 71 | f360_hair_dryer | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 93 | 5 | 4 | 1 | 1 (+1) |
| 72 | f360_kick_scooter | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 40 | 4 | 5 | 1 | 3 (+0) |
| 73 | f360_unicycle | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 38 | 8 | 6 | 2 | 4 (+0) |
| 74 | f360_electric_guitar | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 176 | 12 | 8 | 5 | 6 (+0) |
| 75 | f360_bicycle | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 26 | 12 | 5 | 3 | 10 (+0) |
| 76 | f360_sailboat | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 30 | 10 | 5 | 4 | 5 (+1) |
| 77 | f360_glider | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 50 | 10 | 6 | 2 | 5 (+3) |
| 78 | f360_vacuum_cleaner | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 33 | 10 | 5 | 4 | 5 (+0) |
| 79 | f360_electric_toothbrush | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 95 | 6 | 6 | 4 | 4 (+0) |
| 80 | f360_smartphone | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 110 | 11 | 8 | 4 | 2 (+1) |
| 81 | f360_pipe_wrench | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 26 | 4 | 5 | 2 | 2 (+0) |
| 82 | f360_weight_bench | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 86 | 1 | 5 | 0 | 1 (+1) |
| 83 | f360_worm_gearbox | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 43 | 5 | 2 | 2 | 2 (+0) |
| 84 | f360_tape_dispenser | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 43 | 4 | 5 | 2 | 1 (+0) |
| 85 | f360_electric_motor | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 93 | 7 | 5 | 3 | 4 (+0) |
| 86 | f360_universal_joint | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 20 | 5 | 4 | 1 | 4 (+0) |
| 87 | f360_remote_control | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 68 | 8 | 6 | 3 | 0 (+0) |
| 88 | f360_dust_collector | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 88 | 7 | 6 | 2 | 3 (+0) |
| 89 | bc_airliner | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 252 | 8 | 6 | 3 | 5 (+1) |
| 90 | bc_airport_tug | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 194 | 0 | 0 | 0 | 0 (+0) |
| 91 | bc_box_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 220 | 16 | 6 | 6 | 5 (+0) |
| 92 | bc_boxcar | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 74 | 4 | 4 | 0 | 2 (+0) |
| 93 | bc_bus | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 205 | 16 | 11 | 6 | 7 (+0) |
| 94 | bc_caboose | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 94 | 7 | 3 | 2 | 1 (+0) |
| 95 | bc_carriage | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 128 | 7 | 5 | 3 | 3 (+2) |
| 96 | bc_diesel_locomotive | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 165 | 16 | 5 | 5 | 7 (+0) |
| 97 | bc_dump_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 206 | 10 | 6 | 4 | 6 (+0) |
| 98 | bc_fishing_boat | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 233 | 8 | 6 | 3 | 9 (+0) |
| 99 | bc_food_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 240 | 9 | 4 | 3 | 3 (+0) |
| 100 | bc_garbage_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 211 | 12 | 5 | 4 | 6 (+1) |
| 101 | bc_go_kart | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 75 | 9 | 6 | 4 | 8 (+0) |
| 102 | bc_hopper_car | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 103 | 7 | 5 | 4 | 4 (+0) |
| 103 | bc_limousine | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 182 | 10 | 3 | 3 | 4 (+0) |
| 104 | bc_logging_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 173 | 9 | 6 | 5 | 8 (+0) |
| 105 | bc_mobile_crane | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 283 | 9 | 5 | 3 | 4 (+0) |
| 106 | bc_monster_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 208 | 11 | 6 | 2 | 6 (+0) |
| 107 | bc_motor_grader | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 197 | 10 | 5 | 3 | 3 (+0) |
| 108 | bc_motorhome | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 178 | 18 | 6 | 5 | 4 (+1) |
| 109 | bc_piano | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 126 | 10 | 8 | 3 | 4 (+1) |
| 110 | bc_pickup_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 212 | 18 | 5 | 5 | 6 (+1) |
| 111 | bc_pipe_organ | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 91 | 9 | 6 | 4 | 3 (+0) |
| 112 | bc_police_car | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 229 | 14 | 5 | 4 | 5 (+2) |
| 113 | bc_race_car | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 152 | 12 | 8 | 5 | 5 (+0) |
| 114 | bc_semi_trailer_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 267 | 12 | 3 | 2 | 7 (+1) |
| 115 | bc_ship | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 108 | 11 | 6 | 5 | 5 (+0) |
| 116 | bc_skid_steer_loader | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 82 | 10 | 6 | 2 | 3 (+1) |
| 117 | bc_snowmobile | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 102 | 10 | 7 | 2 | 2 (+0) |
| 118 | bc_steam_locomotive | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 242 | 15 | 5 | 4 | 7 (+0) |
| 119 | bc_submarine | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 176 | 9 | 6 | 4 | 5 (+0) |
| 120 | bc_table_football | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 132 | 6 | 4 | 2 | 3 (+0) |
| 121 | bc_table_saw | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 108 | 9 | 6 | 3 | 4 (+2) |
| 122 | bc_tank | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 98 | 10 | 6 | 4 | 4 (+1) |
| 123 | bc_tank_car | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 113 | 9 | 6 | 3 | 2 (+0) |
| 124 | bc_tank_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 202 | 7 | 5 | 2 | 4 (+0) |
| 125 | bc_tow_truck | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 168 | 15 | 5 | 4 | 9 (+0) |
| 126 | bc_tram | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 123 | 11 | 8 | 4 | 3 (+1) |
| 127 | bc_tricycle | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 113 | 7 | 6 | 2 | 4 (+1) |
| 128 | bc_van | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 161 | 15 | 4 | 3 | 6 (+0) |
| 129 | bc_vending_machine | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 80 | 8 | 4 | 2 | 2 (+1) |
| 130 | bc_auto_rickshaw | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 100 | 10 | 5 | 3 | 2 (+0) |
| 131 | bc_road_roller | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 156 | 6 | 5 | 2 | 3 (+0) |
| 132 | bc_backhoe_loader | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 157 | 11 | 4 | 5 | 5 (+0) |
| 133 | bc_carousel | BrickComposer collection (BrickLink Studio models) | per-model terms of the original Studio model; not established here | 3 | 4 | 266 | 5 | 5 | 2 | 2 (+0) |
| 134 | robotic_hand | Curated real-world CAD sources | (none given) | 3 | 4 | 236 | 8 | 6 | 3 | 4 (+0) |
| 135 | computer_keyboard | Curated real-world CAD sources | (none given) | 3 | 4 | 250 | 8 | 6 | 2 | 1 (+1) |
| 136 | computer_case | Curated real-world CAD sources | (none given) | 3 | 4 | 232 | 9 | 7 | 3 | 1 (+1) |
| 137 | pc_pen_plotter | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-SA-4.0 | 3 | 4 | 65 | 8 | 5 | 3 | 4 (+0) |
| 138 | pc_underwater_rov | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-NC-SA-4.0 | 3 | 4 | 95 | 8 | 6 | 5 | 4 (+0) |
| 139 | pc_raman_spectrometer | Open-source hardware product CAD (fetched 2026-09-27) | CERN-OHL-W-2.0 | 3 | 4 | 285 | 7 | 6 | 1 | 1 (+0) |
| 140 | pc_testing_machine | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-4.0 | 3 | 4 | 125 | 6 | 6 | 2 | 2 (+0) |
| 141 | pc_rheometer | Open-source hardware product CAD (fetched 2026-09-27) | MIT | 3 | 4 | 150 | 8 | 6 | 2 | 5 (+1) |
| 142 | pc_thermal_cycler | Open-source hardware product CAD (fetched 2026-09-27) | GPL-3.0 | 3 | 4 | 188 | 5 | 4 | 2 | 0 (+0) |
| 143 | pc_shredder | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-SA-4.0 | 3 | 4 | 138 | 7 | 7 | 0 | 0 (+0) |
| 144 | pc_injection_moulder | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-SA-4.0 | 3 | 4 | 93 | 10 | 6 | 4 | 5 (+0) |
| 145 | pc_heat_press | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-SA-4.0 | 3 | 4 | 258 | 6 | 6 | 0 | 5 (+0) |
| 146 | pc_equatorial_mount | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-NC-4.0 | 3 | 4 | 88 | 4 | 4 | 0 | 4 (+0) |
| 147 | pc_cubesat | Open-source hardware product CAD (fetched 2026-09-27) | CERN-OHL-1.2 | 3 | 4 | 318 | 10 | 6 | 4 | 4 (+0) |
| 148 | pc_blade_grinder | Open-source hardware product CAD (fetched 2026-09-27) | CERN-OHL-W-2.0 | 3 | 4 | 90 | 3 | 6 | 1 | 1 (+0) |
| 149 | pc_camera_slider | Open-source hardware product CAD (fetched 2026-09-27) | GPL-3.0 | 3 | 4 | 68 | 6 | 7 | 3 | 2 (+1) |
| 150 | pc_3d_scanner | Open-source hardware product CAD (fetched 2026-09-27) | CERN-OHL-W-2.0 | 3 | 4 | 245 | 5 | 6 | 2 | 3 (+0) |
| 151 | f360_calipers | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 38 | 7 | 5 | 3 | 5 (+0) |
| 152 | f360_coilover | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 50 | 8 | 6 | 3 | 3 (+0) |
| 153 | f360_door_lock | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 108 | 6 | 5 | 1 | 7 (+0) |
| 154 | f360_fighter_jet | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 15 | 9 | 4 | 2 | 4 (+1) |
| 155 | f360_fuel_injector | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 56 | 7 | 4 | 2 | 2 (+0) |
| 156 | f360_master_cylinder | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 63 | 8 | 5 | 3 | 2 (+0) |
| 157 | f360_moka_pot | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 33 | 8 | 5 | 3 | 2 (+0) |
| 158 | f360_railway_bogie | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 228 | 7 | 5 | 5 | 3 (+2) |
| 159 | f360_record_player | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 18 | 9 | 5 | 4 | 3 (+0) |
| 160 | f360_pan_tilt_head | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 65 | 6 | 2 | 1 | 3 (+0) |
| 161 | f360_roller_conveyor | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 35 | 6 | 5 | 2 | 2 (+0) |
| 162 | f360_winch | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 163 | 6 | 5 | 3 | 4 (+0) |
| 163 | f360_wristwatch | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 26 | 10 | 4 | 4 | 6 (+2) |
| 164 | f360_motorcycle_fork | Fusion 360 Gallery Assembly Dataset | CC BY-NC-SA 4.0 (Autodesk research licence) | 3 | 4 | 50 | 5 | 5 | 2 | 2 (+0) |
| 165 | av_wardrobe | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 60 | 5 | 3 | 2 | 1 (+0) |
| 166 | av_tv_stand | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 6 | 4 | 2 | 1 (+1) |
| 167 | av_bathroom_vanity | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 77 | 4 | 4 | 2 | 1 (+0) |
| 168 | av_blender | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 24 | 7 | 7 | 3 | 2 (+0) |
| 169 | av_electric_kettle | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 7 | 5 | 3 | 0 (+0) |
| 170 | av_toaster | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 6 | 6 | 1 | 3 (+0) |
| 171 | av_soap_dispenser | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 8 | 5 | 1 | 3 (+1) |
| 172 | av_flashlight | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 6 | 5 | 3 | 3 (+1) |
| 173 | av_bar_stool | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 27 | 3 | 7 | 1 | 1 (+0) |
| 174 | av_recliner | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 8 | 7 | 3 | 4 (+0) |
| 175 | av_spray_bottle | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 6 | 2 | 2 | 0 (+0) |
| 176 | av_safe | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 47 | 6 | 6 | 2 | 4 (+0) |
| 177 | av_pedal_bin | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 20 | 3 | 5 | 1 | 2 (+0) |
| 178 | av_game_controller | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 40 | 8 | 5 | 2 | 3 (+0) |
| 179 | av_locker | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 19 | 5 | 8 | 3 | 1 (+0) |
| 180 | av_standing_mirror | Artiverse | gated; per-object licences inherited from the upstream asset sources | 3 | 4 | 34 | 6 | 6 | 3 | 1 (+1) |
| 181 | pn_bucket | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 100 | 4 | 5 | 2 | 0 (+0) |
| 182 | pn_chandelier | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 90 | 5 | 6 | 0 | 0 (+0) |
| 183 | pn_computer_monitor | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 24 | 8 | 6 | 2 | 3 (+2) |
| 184 | pn_eyeglasses | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 44 | 7 | 5 | 2 | 1 (+1) |
| 185 | pn_backpack | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 20 | 3 | 4 | 1 | 1 (+3) |
| 186 | pn_pen | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 20 | 5 | 2 | 1 | 3 (+0) |
| 187 | pn_teapot | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 20 | 5 | 5 | 3 | 0 (+0) |
| 188 | pn_sword | PartNeXt | CC BY 4.0; assets from Objaverse, ABO and 3D-FUTURE under their own terms | 3 | 4 | 20 | 8 | 6 | 2 | 2 (+0) |
| 189 | pc_3d_printer | Open-source hardware product CAD (fetched 2026-09-27) | GPL-3.0 | 3 | 4 | 473 | 10 | 4 | 3 | 4 (+0) |
| 190 | pc_ventilator | Open-source hardware product CAD (fetched 2026-09-27) | CERN-OHL-S-2.0 | 3 | 4 | 311 | 5 | 8 | 4 | 3 (+0) |
| 191 | pc_peristaltic_pump | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-4.0 | 3 | 4 | 315 | 5 | 6 | 3 | 3 (+0) |
| 192 | pc_vinyl_cutter | Open-source hardware product CAD (fetched 2026-09-27) | CERN-OHL-W-2.0 | 3 | 4 | 350 | 6 | 6 | 4 | 5 (+0) |
| 193 | pc_mechanical_calculator | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-NC-SA-4.0 | 3 | 4 | 327 | 10 | 4 | 4 | 7 (+0) |
| 194 | pc_card_shuffler | Open-source hardware product CAD (fetched 2026-09-27) | MIT | 3 | 4 | 78 | 8 | 5 | 3 | 5 (+0) |
| 195 | pc_wire_stripper | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-NC-4.0 | 3 | 4 | 58 | 4 | 6 | 4 | 6 (+0) |
| 196 | pc_robotic_mower | Open-source hardware product CAD (fetched 2026-09-27) | GPL-3.0 | 3 | 4 | 56 | 10 | 4 | 4 | 5 (+0) |
| 197 | pc_ceb_press | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-SA-4.0 | 3 | 4 | 311 | 1 | 6 | 0 | 2 (+0) |
| 198 | pc_mobile_robot | Open-source hardware product CAD (fetched 2026-09-27) | MIT | 3 | 4 | 301 | 7 | 5 | 2 | 3 (+0) |
| 199 | pc_racing_pedal | Open-source hardware product CAD (fetched 2026-09-27) | CC-BY-NC-SA-4.0 | 3 | 4 | 158 | 1 | 3 | 2 | 1 (+0) |
| 200 | pc_vr_headset | Open-source hardware product CAD (fetched 2026-09-27) | CERN-OHL-S-2.0 | 3 | 4 | 309 | 7 | 5 | 3 | 2 (+1) |
| | **total** | | | 580 | 805 | 20518 | 1523 | 1071 | 534 | 689 (+90) |

## Size

18447 files, 12.07 GB (excluding MANIFEST.json). By subdirectory:

- `(top-level files)`: 21.4 MB
- `assets/artiverse`: 105.5 MB
- `assets/ldraw`: 103.1 MB
- `tasks`: 11843.0 MB

Source datasets by task count: Fusion 360 Gallery Assembly Dataset: 55, BrickComposer collection (BrickLink Studio models): 50, Artiverse: 42, Open-source hardware product CAD (fetched 2026-09-27): 26, PartNeXt: 14, Curated real-world CAD sources: 8, BrickNet corpus (LDraw Official Model Repository derived): 5.
