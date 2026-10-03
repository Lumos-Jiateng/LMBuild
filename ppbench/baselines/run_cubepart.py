#!/usr/bin/env python3
"""Cube3D v0.5 (text -> shape) + CubePart (shape + part schema -> one mesh per part).

Two-stage text-to-part-decomposed-3D baseline. Standalone; imports nothing from ppbench.
Run with the dedicated venv:

    V=third_party/cube/.venv/bin/python
    $V ppbench/baselines/run_cubepart.py --prompt "An office chair" \
        --parts "seat, backrest, armrests, gas lift cylinder, base, casters" \
        --setting short_short --out results/v2/swivel_office_chair/external --seeds 0 1 2

The orchestrator (this process, CPU only) launches every GPU stage as a separate
subprocess wrapped in `flock <lock>` with CUDA_VISIBLE_DEVICES=<gpu>, so only one model
is in GPU memory at a time:
  stage "shape": Cube3D v0.5, official generate.py path (Engine + generate_mesh with
                 pymeshlab cleanup/decimation), top_p=0.9 so that seeds matter.
  stage "parts": CubePart, settings of cubepart/examples/run_inference.py
                 (guidance 7.5, 50 steps, dpm_solver, timeshift 4.0, resolution 8.5,
                 128k surface samples). The Qwen3-VL text encoder, the DiT and the VAE
                 are moved to the GPU one after another.

Outputs per run: <out>/cubepart_<setting>_s<seed>/{shape.glb, shape.obj,
parts/<idx>_<name>.glb, parts_scene.glb, meta.json}.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUBE = Path(os.environ.get("PPBENCH_CUBE", ROOT / "third_party/cube"))
HF_HOME = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
LOCK = os.environ.get("PPBENCH_GPU_LOCK", "/tmp/ppbench_gpu.lock")
TMP = "/tmp/ppbench_cube"
OFFICIAL_MAX_PARTS = 8  # cubepart/examples/gradio_demo.py MAX_PARTS; pipeline hard-codes 8 slots

SHAPE_CFG = dict(repo="Roblox/cube3d-v0.5", config="cube3d/configs/open_model_v0.5.yaml",
                 guidance_scale=3.0, resolution_base=8.0, top_p=0.9, use_kv_cache=True,
                 bounding_box_xyz=None, postprocess=True, fast_inference=False,
                 gpt_dtype="bf16 (Linear/Embedding; norms, text_proj, bbox_proj kept fp32)",
                 shape_tokenizer_dtype="fp32", clip_dtype="fp32 (official: full precision)")
PARTS_CFG = dict(repo="Roblox/cubepart", config="configs/shape_denoiser_multimesh.yaml",
                 guidance_scale=7.5, num_inference_steps=50, scheduler_type="dpm_solver",
                 timeshift=4.0, resolution_base=8.5, chunk_size=100_000, num_samples=128_000,
                 extract_geometry_fn_name="extract_geometry_coarse_to_fine", mesh_scale=0.96,
                 text_encoder="Qwen/Qwen3-VL-4B-Instruct (fp16)",
                 dit_dtype="bf16 (Linear; norms and time_text_embed kept fp32)", vae_dtype="fp32")


def snapshot(repo):
    d = Path(HF_HOME) / "hub" / ("models--" + repo.replace("/", "--"))
    try:
        rev = (d / "refs/main").read_text().strip()
    except OSError:
        return None, None
    return rev, d / "snapshots" / rev


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def jdump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str))


def gpu_stats(torch):
    return dict(peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 3),
                device_name=torch.cuda.get_device_name(0))


def seed_all(seed):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def half_linears_(module, torch, keep_fp32=()):
    """Cast nn.Linear / nn.Embedding weights to bf16 except modules whose name starts with keep_fp32.

    Norm layers stay fp32 because both repos' LayerNorm wrappers call F.layer_norm on
    input.float(); modules called outside autocast (see keep_fp32) must match fp32 inputs.
    """
    n = 0
    for name, m in module.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)) and not any(
                name == k or name.startswith(k + ".") for k in keep_fp32):
            m.to(torch.bfloat16); n += 1
    return n


# ----------------------------------------------------------------------------- frame check
def frame_analysis(mesh, np):
    """Check the +Y up / +Z forward convention CubePart expects, for a chair-like object.

    Up: the tallest bbox axis should be Y. Forward: the top 30% of the height is mostly
    backrest, which should sit on the -Z side of the bbox centre (seat opens towards +Z).
    Returns (report, 4x4 transform to apply, or None if identity).
    """
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(0), v.max(0); ext = hi - lo; ctr = (lo + hi) / 2
    rep = dict(bbox_min=lo.tolist(), bbox_max=hi.tolist(), extents_xyz=ext.tolist(),
               tallest_axis="XYZ"[int(ext.argmax())])
    T = np.eye(4)
    if rep["tallest_axis"] != "Y":
        a = int(ext.argmax())
        # rotate the tallest axis onto +Y by 90 deg about the remaining axis (sign kept simple)
        R = np.eye(3)
        if a == 2:  # Z up -> Y up : rotate -90 about X
            R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        elif a == 0:  # X up -> Y up : rotate +90 about Z
            R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        T[:3, :3] = R
        v = v @ R.T
        lo, hi = v.min(0), v.max(0); ext = hi - lo; ctr = (lo + hi) / 2
        rep["up_fix"] = f"rotated tallest axis {'XYZ'[a]} onto Y"
    top = v[v[:, 1] > hi[1] - 0.3 * ext[1]]
    off = (top.mean(0) - ctr) if len(top) else np.zeros(3)
    rep["top30pct_centroid_offset_xz"] = [float(off[0]), float(off[2])]
    thr = 0.05 * ext.max()
    if abs(off[2]) >= abs(off[0]) and off[2] < -thr:
        rep["forward"] = "+Z (backrest mass at -Z); no rotation"
    elif abs(off[2]) >= abs(off[0]) and off[2] > thr:
        R = np.diag([-1.0, 1.0, -1.0]); T = np.diag([-1.0, 1.0, -1.0, 1.0]) @ T
        rep["forward"] = "backrest mass at +Z -> rotated 180 deg about Y"
    elif abs(off[0]) > abs(off[2]) and abs(off[0]) > thr:
        s = np.sign(off[0])  # backrest at s*X; rotate about Y so it lands on -Z
        # rotation about Y by angle t maps (x,z)->(x cos t + z sin t, -x sin t + z cos t);
        # want (s,0)->(0,-1): t = +90 deg if s>0 else -90 deg
        t = np.pi / 2 * s
        Ry = np.array([[np.cos(t), 0, np.sin(t), 0], [0, 1, 0, 0], [-np.sin(t), 0, np.cos(t), 0], [0, 0, 0, 1]])
        T = Ry @ T
        rep["forward"] = f"backrest mass on {'+' if s > 0 else '-'}X -> rotated {int(np.degrees(t))} deg about Y"
    else:
        rep["forward"] = "ambiguous (top-region centroid near bbox centre); left unchanged"
    ident = np.allclose(T, np.eye(4))
    return rep, (None if ident else T)


# ----------------------------------------------------------------------------- stage 1
def stage_shape(a, run):
    import numpy as np, torch, trimesh
    sys.path.insert(0, str(CUBE))
    os.chdir(CUBE)  # config paths in cube3d are repo-relative
    from cube3d.inference.engine import Engine
    from cube3d.generate import generate_mesh
    from cube3d.mesh_utils.postprocessing import PYMESHLAB_AVAILABLE
    meta = jload(run / "meta.json"); warns = meta.setdefault("warnings", [])
    rev, snap = snapshot(SHAPE_CFG["repo"])
    t0 = time.time()
    dev = torch.device("cuda")
    eng = Engine(SHAPE_CFG["config"], str(snap / "shape_gpt.safetensors"),
                 str(snap / "shape_tokenizer.safetensors"), device=torch.device("cpu"))
    nh = half_linears_(eng.gpt_model, torch, keep_fp32=("text_proj", "bbox_proj"))
    eng.device = dev
    eng.gpt_model.to(dev); eng.shape_model.to(dev); eng.text_model.to(dev)
    t_load = time.time() - t0
    seed_all(a.seed)
    t1 = time.time()
    work = run / "_stage1"; work.mkdir(exist_ok=True)
    obj = generate_mesh(eng, a.prompt, str(work), "output", SHAPE_CFG["resolution_base"],
                        not SHAPE_CFG["postprocess"], SHAPE_CFG["top_p"], None)
    t_gen = time.time() - t1
    if not PYMESHLAB_AVAILABLE:
        warns.append("pymeshlab unavailable: official post-processing skipped")
    m = trimesh.load(obj, force="mesh", process=False)
    shutil.copy(obj, run / "shape.obj")
    m.export(run / "shape.glb")
    rep, T = frame_analysis(m, np)
    if T is not None:
        m2 = m.copy(); m2.apply_transform(T); m2.export(run / "shape_aligned.glb")
        rep["cubepart_input"] = "shape_aligned.glb"; rep["transform_shape_to_aligned"] = T.tolist()
        warns.append("stage-1 mesh was not in +Y up/+Z forward; rotated copy fed to CubePart: " + json.dumps({k: rep[k] for k in rep if k in ('up_fix', 'forward')}))
    else:
        rep["cubepart_input"] = "shape.glb (identity; Cube3D output already +Y up, +Z forward)"
    shutil.rmtree(work, ignore_errors=True)
    meta["stage1_shape"] = dict(
        model="Cube3D v0.5 text-to-shape", weights=dict(repo=SHAPE_CFG["repo"], revision=rev,
        clip="openai/clip-vit-large-patch14@" + str(snapshot("openai/clip-vit-large-patch14")[0])),
        settings=dict(SHAPE_CFG, n_linear_cast_bf16=nh, seed_mechanism="random/np/torch.manual_seed before t2s; top_p sampling via torch.multinomial"),
        wall_s=dict(load=round(t_load, 1), generate_and_postprocess=round(t_gen, 1), total=round(time.time() - t0, 1)),
        gpu=gpu_stats(torch), mesh=dict(n_vertices=int(len(m.vertices)), n_faces=int(len(m.faces)),
        watertight=bool(m.is_watertight), n_components=int(len(m.split(only_watertight=False)))),
        frame=rep)
    jdump(run / "meta.json", meta)
    print("stage shape done", json.dumps(meta["stage1_shape"]["wall_s"]), rep)


# ----------------------------------------------------------------------------- stage 2
def stage_parts(a, run):
    import numpy as np, torch, trimesh, colorsys
    sys.path.insert(0, str(CUBE / "cubepart"))
    os.chdir(CUBE / "cubepart")
    from cube_part.utils.config import load_config
    from cube_part.systems.shape_denoiser import ShapeDenoiserSystem
    from cube_part.pipelines import PartShapeDenoiserPipeline, ShapeInput
    from cube_part.utils.mesh import load_mesh, sample_surface
    from safetensors import safe_open
    meta = jload(run / "meta.json"); warns = meta.setdefault("warnings", [])
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    num_slots = max(OFFICIAL_MAX_PARTS, len(parts))
    if len(parts) > OFFICIAL_MAX_PARTS:
        warns.append(f"schema has {len(parts)} parts > official maximum {OFFICIAL_MAX_PARTS} "
                     f"(gradio demo MAX_PARTS, pipeline hard-codes 8 slots); ran with num_parts={num_slots} "
                     "slots via the PP_Bench patch -- out of the released operating range")
    rev, snap = snapshot(PARTS_CFG["repo"])
    dev = torch.device("cuda")
    t0 = time.time()
    cfg = load_config(PARTS_CFG["config"])
    cfg.system.pretrained_model_path = str(snap / "multi_part_dit.safetensors")
    cfg.system.shape_model.pretrained_model_path = str(snap / "vae.safetensors")
    cfg.system.attn_implementation = "sdpa"; cfg.system.gradient_checkpointing = False
    torch.set_grad_enabled(False)
    system = ShapeDenoiserSystem(cfg.system).eval()  # everything on CPU first
    with safe_open(cfg.system.pretrained_model_path, "pt") as f:
        ck = set(f.keys())
    sd = set(system.state_dict().keys())
    missing = sorted(k for k in sd - ck if not k.startswith(("base_model.", "shape_model.")))
    unexpected = sorted(ck - sd)
    if missing or unexpected:
        warns.append(f"DiT checkpoint key mismatch: missing={missing[:5]}({len(missing)}) unexpected={unexpected[:5]}({len(unexpected)})")
    pipe = PartShapeDenoiserPipeline.__new__(PartShapeDenoiserPipeline)
    pipe.system, pipe.device, pipe.cfg = system, dev, cfg
    pipe.extract_geometry_fn_name = PARTS_CFG["extract_geometry_fn_name"]
    system.shape_model.to(dev)
    system.shape_model_shift.data = system.shape_model_shift.data.to(dev)
    system.shape_model_scale.data = system.shape_model_scale.data.to(dev)
    t_load = time.time() - t0
    wall = dict(load_cpu=round(t_load, 1))

    # 1) VAE encode of the stage-1 mesh (normalised by load_mesh to [-0.96, 0.96])
    src = run / ("shape_aligned.glb" if (run / "shape_aligned.glb").exists() else "shape.glb")
    t = time.time()
    mesh, center, scale = load_mesh(str(src))
    seed_all(a.seed)
    surf = sample_surface(mesh, num_samples=PARTS_CFG["num_samples"])
    surf = torch.from_numpy(surf).to(dev).unsqueeze(0).float()
    latents, _ = pipe.encode_shape(surf)
    wall["vae_encode"] = round(time.time() - t, 1)
    mem = {"after_vae_encode": gpu_stats(torch)}

    # 2) text encoder on GPU alone; prompts built exactly as input_to_part_shape builds them
    t = time.time()
    padded = parts + [""] * (num_slots - len(parts))
    texts = system.apply_part_text_template([padded])
    texts = texts + [system.default_negative_prompt] * len(texts)  # guidance_scale > 0
    real = system.base_model
    real.text_encoder.to(dev)
    ehs, emask = real(texts)
    real.text_encoder.to("cpu")

    class CachedText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.processor, self.prompt_template_encode = real.processor, real.prompt_template_encode

        def forward(self, text):
            assert list(text) == texts, "prompt mismatch vs cached text encoding"
            return ehs, emask

    system.base_model = CachedText(); del real
    torch.cuda.empty_cache()
    wall["text_encode"] = round(time.time() - t, 1)
    mem["after_text_encode"] = gpu_stats(torch)

    # 3) DiT in bf16 on GPU, sample part latents
    t = time.time()
    dm = system.diffusion_model
    nh = half_linears_(dm, torch, keep_fp32=("time_text_embed",))
    dm.to(dev)
    part_lat = pipe.input_to_part_shape(
        ShapeInput(prompt=[parts], latents=latents), guidance_scale=PARTS_CFG["guidance_scale"],
        resolution_base=PARTS_CFG["resolution_base"], chunk_size=PARTS_CFG["chunk_size"], seed=a.seed,
        scheduler_type=PARTS_CFG["scheduler_type"], timeshift=PARTS_CFG["timeshift"],
        num_inference_steps=PARTS_CFG["num_inference_steps"], output_mesh=False, num_parts=num_slots)
    dm.to("cpu"); torch.cuda.empty_cache()
    wall["diffusion"] = round(time.time() - t, 1)
    mem["after_diffusion"] = gpu_stats(torch)

    # 4) VAE decode -> one mesh per schema element (same call as input_to_part_shape)
    t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        meshes = pipe.decode_shape(part_lat.float(), resolution_base=PARTS_CFG["resolution_base"],
                                   chunk_size=PARTS_CFG["chunk_size"])
    wall["vae_decode"] = round(time.time() - t, 1)
    mem["peak"] = gpu_stats(torch)

    out = run / "parts"
    shutil.rmtree(out, ignore_errors=True); out.mkdir()
    scene = trimesh.Scene(); records, empty = [], []
    for i, name in enumerate(parts):
        v, f = meshes[i] if i < len(meshes) else (None, None)
        rec = dict(index=i, name=name)
        if v is None or f is None or len(v) == 0 or len(f) == 0:
            rec["empty"] = True; empty.append(name)
        else:
            fn = f"{i:02d}_{name.replace(' ', '_')}.glb"
            pm = trimesh.Trimesh(v, f, process=False); pm.export(out / fn)
            lo, hi = pm.bounds
            rec.update(file=f"parts/{fn}", empty=False, n_vertices=int(len(v)), n_faces=int(len(f)),
                       bbox_min=lo.tolist(), bbox_max=hi.tolist(),
                       n_components=int(len(pm.split(only_watertight=False))))
            c = pm.copy()
            r, g, b = colorsys.hsv_to_rgb(i / len(parts), 0.55, 0.95)
            c.visual.face_colors = [int(r * 255), int(g * 255), int(b * 255), 255]
            scene.add_geometry(c, geom_name=f"{i:02d}_{name}")
        records.append(rec)
    if len(scene.geometry):
        scene.export(run / "parts_scene.glb")
    if empty:
        warns.append(f"empty parts: {empty}")
    meta["stage2_parts"] = dict(
        model="CubePart multi-part DiT + shape VAE",
        weights=dict(repo=PARTS_CFG["repo"], revision=rev,
                     text_encoder="Qwen/Qwen3-VL-4B-Instruct@" + str(snapshot("Qwen/Qwen3-VL-4B-Instruct")[0]),
                     dit_config="Qwen/Qwen-Image transformer/config.json@" + str(snapshot("Qwen/Qwen-Image")[0])),
        input_mesh=src.name, schema=parts, num_part_slots=num_slots,
        settings=dict(PARTS_CFG, n_linear_cast_bf16=nh, seed_mechanism="random/np/torch seeds before surface sampling; pipeline generator seed for initial noise"),
        wall_s=dict(wall, total=round(time.time() - t0, 1)), gpu=mem,
        frame=dict(convention="CubePart native: +Y up, +Z forward, mesh normalised by load_mesh to max half-extent 0.96 (glTF units = that normalised space)",
                   to_input_mesh_frame="v_input = v_part / scale + center",
                   scale=float(scale), center=np.asarray(center).tolist()),
        parts=records, empty_parts=empty)
    jdump(run / "meta.json", meta)
    print("stage parts done", json.dumps(wall), "empty:", empty)


# ----------------------------------------------------------------------------- orchestrator
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--parts", required=True, help="comma-separated part schema")
    ap.add_argument("--out", required=True, help="parent dir; runs go to <out>/cubepart_<setting>_s<seed>")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--setting", default="custom", help="run-name tag")
    ap.add_argument("--shape-from", default=None, help="reuse stage-1 output from this run-dir template, e.g. '<out>/cubepart_short_full_s{seed}'")
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--lock", default=LOCK)
    ap.add_argument("--stage", choices=["all", "shape", "parts"], default="all")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--run", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._worker:
        run = Path(a.run)
        (stage_shape if a.stage == "shape" else stage_parts)(a, run)
        return

    out = Path(a.out).resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, HF_HOME=HF_HOME, HF_HUB_OFFLINE="1", TMPDIR=TMP,
               TORCHINDUCTOR_CACHE_DIR=f"{TMP}/inductor", TRITON_CACHE_DIR=f"{TMP}/triton",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    Path(TMP).mkdir(parents=True, exist_ok=True)
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    for seed in a.seeds:
        run = out / f"cubepart_{a.setting}_s{seed}"; run.mkdir(parents=True, exist_ok=True)
        import torch, transformers, diffusers, trimesh, packaging  # noqa: versions only
        meta = dict(system="Cube3D v0.5 (text->shape) + CubePart (shape+schema->parts)", setting=a.setting,
                    prompt=a.prompt, schema=parts, seed=seed, gpu=f"CUDA_VISIBLE_DEVICES={a.gpu}",
                    code=dict(cube_repo=str(CUBE), upstream_commit=(CUBE / "UPSTREAM_COMMIT").read_text().strip()
                              if (CUBE / "UPSTREAM_COMMIT").exists() else None,
                              local_patch="cubepart/cube_part/pipelines/shape_denoiser.py: num_parts kwarg (default 8 = upstream); see third_party/cube/PPBENCH_PATCH.diff"),
                    env=dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
                             diffusers=diffusers.__version__, trimesh=trimesh.__version__,
                             packaging=packaging.__version__ + " (pinned: 26.x breaks torch 2.8 dynamo tracing of diffusers is_torch_version inside the compiled DiT)"),
                    frame_convention="glTF +Y up, +Z forward (CubePart canonical)", warnings=[],
                    reproduce=" ".join([sys.executable, str(Path(__file__).resolve()), "--prompt", json.dumps(a.prompt),
                                        "--parts", json.dumps(a.parts), "--out", str(out), "--setting", a.setting,
                                        "--seeds", str(seed)] + (["--shape-from", json.dumps(a.shape_from)] if a.shape_from else [])))
        old = jload(run / "meta.json")
        if a.stage == "parts" and "stage1_shape" in old:
            meta["stage1_shape"] = old["stage1_shape"]
            meta["warnings"] = [w for w in old.get("warnings", []) if w.startswith("stage-1")]
        jdump(run / "meta.json", meta)
        stages = []
        if a.stage in ("all", "shape"):
            if a.shape_from:
                src = Path(a.shape_from.format(seed=seed))
                for fn in ("shape.glb", "shape.obj", "shape_aligned.glb"):
                    if (src / fn).exists():
                        shutil.copy(src / fn, run / fn)
                m = jload(src / "meta.json"); meta["stage1_shape"] = dict(m["stage1_shape"], reused_from=str(src))
                meta["warnings"] += [w for w in m.get("warnings", []) if w.startswith("stage-1")]
                jdump(run / "meta.json", meta)
            else:
                stages.append("shape")
        if a.stage in ("all", "parts"):
            stages.append("parts")
        for st in stages:
            cmd = ["flock", a.lock, sys.executable, str(Path(__file__).resolve()), "--_worker", "--stage", st,
                   "--seed", str(seed), "--run", str(run), "--prompt", a.prompt, "--parts", a.parts, "--out", str(out)]
            t = time.time()
            with open(run / f"log_{st}.txt", "w") as lf:
                rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            m = jload(run / "meta.json"); m.setdefault("process_wall_s", {})[st] = round(time.time() - t, 1)
            if rc != 0:
                m.setdefault("failures", []).append(f"stage {st} exited {rc}; see log_{st}.txt")
            jdump(run / "meta.json", m)
            print(f"[seed {seed}] stage {st} rc={rc} ({time.time() - t:.0f}s) -> {run}", flush=True)
            if rc != 0:
                break


if __name__ == "__main__":
    main()
