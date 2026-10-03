#!/usr/bin/env python3
"""PhysX-Anything (image -> sim-ready articulated asset: URDF + MJCF + per-part meshes + physics JSON).

Standalone. The driver uses only the stdlib and can be started from any interpreter; every stage runs in
third_party/physx_anything/.venv by re-invoking this file with --_stage. The four upstream scripts
(1_vlm_demo.py, 2_decoder.py, 3_split.py, 4_simready_gen.py) run unmodified through runpy, with their
hard-coded relative paths (./demo, ./test_demo, ./pretrain, ./dataset, mjcf_source/) satisfied by a
per-run work directory. Two things are injected by monkeypatching, and both are recorded in meta.json:
  * VLM precision: Qwen2.5-VL-7B bf16 (~16.6 GB of weights) does not fit on a shared ~17 GB GPU, so the
    language model is loaded with bitsandbytes int8 (vision tower and lm_head stay bf16); nf4 is used
    only if int8 runs out of memory.
  * Decoder seed: 2_decoder.py hard-codes seed=1; run_control() is patched to use --seeds.
The VLM stage decodes greedily (do_sample=False), so it does not depend on the seed. It runs once per
image and is shared by all seeds unless --vlm-per-seed is given. Seeds change only the TRELLIS-based
decoder, and through it the mesh split and the per-part meshes.

    python ppbench/baselines/run_physx_anything.py --image IMG --out results/v2/<task>/external --seeds 0 1 2
"""
import argparse, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

PPBENCH = Path(__file__).resolve().parents[2]
PX = PPBENCH / "third_party/physx_anything"
REPO = PX / "repo"
VENV_PY = PX / ".venv/bin/python"
LOCK = os.environ.get("PPBENCH_GPU_LOCK", "/tmp/ppbench_gpu.lock")
HF_HOME = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
NAME = "input"  # file stem under ./demo; upstream scripts name the result dir test_demo/<stem>


# --------------------------------------------------------------------------- stage side (runs in .venv)
def _gpu_poller(stats):
    """Record this process's peak memory as seen by nvidia-smi (includes CUDA context)."""
    pid = str(os.getpid())
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
            for line in out.splitlines():
                p, m = [x.strip() for x in line.split(",")]
                if p == pid:
                    stats["nvml_peak_mib"] = max(stats.get("nvml_peak_mib", 0), int(m))
        except Exception:
            pass
        time.sleep(2)


def run_stage(stage, work, stats_path, precision, seed, extra):
    import runpy
    os.chdir(work)
    sys.path.insert(0, str(REPO))
    stats = {"stage": stage}
    t0 = time.time()
    if stage in ("vlm", "decoder"):
        threading.Thread(target=_gpu_poller, args=(stats,), daemon=True).start()
    try:
        if stage == "vlm":
            import warnings
            import torch
            # bitsandbytes emits this on every int8 matmul (120 MB of log per image otherwise)
            warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast")
            from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
            orig = Qwen2_5_VLForConditionalGeneration.from_pretrained.__func__

            def patched(cls, *a, **kw):
                if precision == "int8":
                    kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True, llm_int8_skip_modules=["visual", "lm_head"])
                elif precision == "nf4":
                    kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_skip_modules=["visual", "lm_head"])
                kw["device_map"] = {"": 0}
                return orig(cls, *a, **kw)
            Qwen2_5_VLForConditionalGeneration.from_pretrained = classmethod(patched)
            torch.manual_seed(seed)
            # argparse type=bool upstream: any non-empty string is True, so omit a flag to get False.
            argv = ["1_vlm_demo.py", "--demo_path", "./demo", "--save_part_ply", "True", "--ckpt", "./pretrain/vlm"]
            if extra.get("remove_bg"):
                argv += ["--remove_bg", "True"]
            sys.argv = argv
            runpy.run_path(str(REPO / "1_vlm_demo.py"), run_name="__main__")
        elif stage == "decoder":
            import torch
            from trellis.pipelines import TrellisImageTo3DPipeline
            orig_rc = TrellisImageTo3DPipeline.run_control

            def run_control(self, pointinput, image, num_samples=1, seed_=None, **kw):
                kw.pop("seed", None)
                return orig_rc(self, pointinput, image, num_samples=num_samples, seed=seed, **kw)
            TrellisImageTo3DPipeline.run_control = run_control
            sys.argv = ["2_decoder.py"]
            runpy.run_path(str(REPO / "2_decoder.py"), run_name="__main__")
        elif stage == "split":
            sys.argv = ["3_split.py"]
            runpy.run_path(str(REPO / "3_split.py"), run_name="__main__")
        elif stage == "simready":
            sys.argv = ["4_simready_gen.py", "--voxel_define", "32", "--basepath", "./test_demo",
                        "--process", str(extra.get("process", 0)), "--fixed_base", str(extra.get("fixed_base", 0)),
                        "--deformable", str(extra.get("deformable", 0))]
            runpy.run_path(str(REPO / "4_simready_gen.py"), run_name="__main__")
        stats["ok"] = True
    except BaseException as e:  # record, then re-raise so the return code reflects the failure
        stats["ok"] = False
        stats["error"] = f"{type(e).__name__}: {e}"[:2000]
        stats["oom"] = "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"
        raise
    finally:
        stats["wall_s"] = round(time.time() - t0, 1)
        if stage in ("vlm", "decoder"):
            try:
                import torch
                stats["torch_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
                stats["torch_max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
            except Exception:
                pass
        Path(stats_path).write_text(json.dumps(stats, indent=1))


# --------------------------------------------------------------------------- driver side
def _git(*args, cwd=REPO):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _snapshot(repo_id):
    ref = Path(HF_HOME) / "hub" / ("models--" + repo_id.replace("/", "--")) / "refs/main"
    return ref.read_text().strip() if ref.exists() else None


def make_work(work, image):
    (work / "demo").mkdir(parents=True, exist_ok=True)
    (work / "test_demo").mkdir(exist_ok=True)
    for name, target in [("dataset", REPO / "dataset"), ("mjcf_source", REPO / "mjcf_source"),
                         ("pretrain", PX / "pretrain")]:
        link = work / name
        if not link.exists():
            link.symlink_to(target)
    shutil.copy(image, work / "demo" / f"{NAME}.png")


def call_stage(stage, work, log_dir, gpu, precision=None, seed=0, extra=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HOME=HF_HOME, HF_HUB_OFFLINE="1",
               TMPDIR=os.environ.get("TMPDIR", "/tmp/ppbench_physx"), SPCONV_ALGO="native",
               CUDA_HOME=CUDA_HOME, PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    stats_path = log_dir / f"{stage}_stats.json"
    cmd = [str(VENV_PY), str(Path(__file__).resolve()), "--_stage", stage, "--_work", str(work),
           "--_stats", str(stats_path), "--_precision", str(precision), "--_seed", str(seed),
           "--_extra", json.dumps(extra or {})]
    if stage in ("vlm", "decoder"):
        # one lock per card (2026-09-18): a single gpu4.lock serialised the two PhysX halves on GPUs 5 and 7,
        # leaving one card idle and unheld while it waited for the other
        cmd = ["flock", LOCK.replace("gpu4", f"gpu{gpu}")] + cmd
    t0 = time.time()
    with open(log_dir / f"{stage}.log", "w") as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {"ok": False}
    stats["returncode"] = rc
    stats["wall_s_with_lock_wait"] = round(time.time() - t0, 1)
    return stats


def run_vlm(work, log_dir, gpu, precision, remove_bg):
    tries = [precision] + (["nf4"] if precision == "int8" else [])
    attempts = []
    for p in tries:
        shutil.rmtree(work / "test_demo" / NAME, ignore_errors=True)
        st = call_stage("vlm", work, log_dir, gpu, precision=p, extra={"remove_bg": remove_bg})
        st["precision"] = p
        attempts.append(st)
        if st.get("ok") and st["returncode"] == 0:
            break
        if not st.get("oom"):
            break
    return attempts


def summarize(res):
    s = {}
    j = res / "basic_info.json"
    if j.exists():
        d = json.loads(j.read_text())
        s["object_name"], s["category"], s["dimension_cm"] = d.get("object_name"), d.get("category"), d.get("dimension")
        s["n_parts"] = len(d.get("parts", []))
        g = d.get("group_info", {})
        s["n_groups"] = len(g)
        s["group_types"] = {k: (v[-1] if k != "0" else "base") for k, v in g.items()}
        s["materials"] = {f'l_{p["label"]}_{p["name"]}': [p["material"], p["density"]] for p in d.get("parts", [])}
    s["n_part_meshes"] = len(list((res / "objs").glob("*/*.obj"))) if (res / "objs").exists() else 0
    for f in ["basic_info.txt", "sample.glb", "basic_info.json", "basic.urdf", "basic.xml"]:
        s["has_" + f] = (res / f).exists()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--out", help="parent dir; writes <tag>_s<seed>/ inside")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--tag", default="physx-anything")
    ap.add_argument("--vlm-precision", default="int8", choices=["bf16", "int8", "nf4"])
    ap.add_argument("--remove-bg", type=int, default=1, help="README: 1 for non-RGBA photos")
    ap.add_argument("--vlm-per-seed", action="store_true")
    ap.add_argument("--work", default=None, help="work dir (default third_party/physx_anything/work/<stem>)")
    ap.add_argument("--_stage"); ap.add_argument("--_work"); ap.add_argument("--_stats")
    ap.add_argument("--_precision"); ap.add_argument("--_seed", type=int, default=0); ap.add_argument("--_extra", default="{}")
    a = ap.parse_args()
    if a._stage:
        return run_stage(a._stage, a._work, a._stats, a._precision, a._seed, json.loads(a._extra))

    image = Path(a.image).resolve()
    out = Path(a.out).resolve()
    work_root = Path(a.work).resolve() if a.work else PX / "work" / f"{image.parent.name}__{image.stem}"
    settings = {"vlm_precision_requested": a.vlm_precision, "remove_bg": bool(a.remove_bg),
                "save_part_ply": True, "vlm_decoding": "greedy (do_sample=False), max_length=32768",
                "vlm_image": "resized to 512x512 (LANCZOS); processor min/max_pixels 65536/262144",
                "decoder": json.loads((PX / "pretrain/decoder/pipeline.json").read_text())["args"]["sparse_structure_sampler"]["params"],
                "decoder_slat_sampler": json.loads((PX / "pretrain/decoder/pipeline.json").read_text())["args"]["slat_sampler"]["params"],
                "glb": {"simplify": 0.5, "texture_size": 1024},
                "simready": {"voxel_define": 32, "process": 0, "fixed_base": 0, "deformable": 0}}
    base_meta = {
        "system": "physx-anything", "paper": "arXiv:2511.13648 (CVPR 2026)",
        "repo": "https://github.com/ziangcao0312/PhysX-Anything", "repo_commit": _git("rev-parse", "HEAD"),
        "repo_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "weights": {"Caoza/PhysX-Anything": _snapshot("Caoza/PhysX-Anything"),
                    "microsoft/TRELLIS-image-large": _snapshot("microsoft/TRELLIS-image-large"),
                    "Qwen/Qwen2.5-VL-7B-Instruct (processor only)": _snapshot("Qwen/Qwen2.5-VL-7B-Instruct"),
                    "dinov2_vitl14_reg (torch.hub facebookresearch/dinov2)": "dinov2_vitl14_reg4_pretrain.pth",
                    "rembg": "u2net.onnx"},
        "image": str(image), "settings": settings, "interpreter": str(VENV_PY),
        "cuda_visible_devices": a.gpu, "command": sys.argv,
        "output_convention": {
            "basic_info.txt": "raw VLM answer: Name/Category/Dimension (cm), parts l_i (name, affordance rank, material, density g/cm^3, Young's modulus GPa, Poisson), group_j joints",
            "basic_info.json": "parsed basic_info.txt: parts[].{label,name,material,density,priority_rank,Basic_description,Young's Modulus (GPa),Poisson's Ratio}; group_info{'0': [base part labels], j: [child labels, parent group id, params(8 or 16), type]}",
            "group_info params": "type C: [axis(3), axis position(3, normalized = voxel/32-0.5), range(2, degrees/180 i.e. multiples of pi)]; type B: [dir(3), position(3), range(2, voxel/32 i.e. normalized length)]; CB: [axis(3), pos(3), rev range(2)/180, slide dir(3), 0 0 0, slide range(2)/32]; A=free, D=ball about point, E=fixed",
            "basic.urdf": "links l_<label> with visual mesh ./objs/<k>/<k>.obj scale 1 (normalized units), mass 1.0 placeholders; articulations via abstract_* links with prismatic/revolute/continuous joints (axis, origin, limit in rad or normalized length)",
            "basic.xml": "MuJoCo MJCF: gravity 0 0 -9.81 (Z-up, metres); mesh scale = max(Dimension cm)/100 so geometry in metres; per-part geom density kg/m^3 (VLM g/cm^3 * 1000); slide/hinge/ball joints on grouppart_<j> bodies",
            "frame": "part meshes: TRELLIS GLB (Y-up) rotated +90 deg about X in 3_split.py -> Z-up, object normalized to the unit cube [-0.5,0.5]^3 (same frame as voxel/32-0.5); real size only enters via MJCF scale",
            "sample.glb": "TRELLIS-decoded whole textured mesh, native TRELLIS GLB frame (Y-up), unit cube, unsegmented",
            "ind_<i>.npy / coord_<i>.txt": "VLM per-part 32^3 voxel coordinates (x,y,z integer) / raw run-length token string",
        },
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    shared_vlm = None
    if not a.vlm_per_seed:
        vwork = work_root / "vlm"
        vlog = vwork / "logs"; vlog.mkdir(parents=True, exist_ok=True)
        make_work(vwork, image)
        shared_vlm = (vwork, run_vlm(vwork, vlog, a.gpu, a.vlm_precision, a.remove_bg))

    for seed in a.seeds:
        warnings = []
        swork = work_root / f"s{seed}"
        shutil.rmtree(swork, ignore_errors=True)
        log_dir = swork / "logs"; log_dir.mkdir(parents=True)
        make_work(swork, image)
        t0 = time.time()
        stages = {}
        if a.vlm_per_seed:
            vwork, attempts = swork, run_vlm(swork, log_dir, a.gpu, a.vlm_precision, a.remove_bg)
        else:
            vwork, attempts = shared_vlm
            warnings.append("VLM stage shared across seeds (greedy decoding is seed-independent); "
                            "its wall_s/peak memory are the single shared run")
            if (vwork / "test_demo" / NAME).exists():
                shutil.copytree(vwork / "test_demo" / NAME, swork / "test_demo" / NAME)
            for f in (vwork / "logs").glob("vlm*"):
                shutil.copy(f, log_dir / f.name)
        stages["vlm_attempts"] = attempts
        vlm_ok = bool(attempts) and attempts[-1].get("ok") and attempts[-1]["returncode"] == 0
        if vlm_ok and attempts[-1]["precision"] != "bf16":
            warnings.append(f"VLM run with bitsandbytes {attempts[-1]['precision']} instead of upstream bf16 "
                            "(GPU memory limit); vision tower and lm_head kept bf16")
        if vlm_ok:
            for st in ["decoder", "split", "simready"]:
                stages[st] = call_stage(st, swork, log_dir, a.gpu, seed=seed)
                if not (stages[st].get("ok") and stages[st]["returncode"] == 0):
                    warnings.append(f"stage {st} failed: {stages[st].get('error')}")
                    break
        else:
            warnings.append(f"VLM stage failed: {attempts[-1].get('error') if attempts else 'not run'}")
        res = swork / "test_demo" / NAME
        dest = out / f"{a.tag}_s{seed}"
        if dest.exists():
            shutil.rmtree(dest)
        if res.exists():
            shutil.copytree(res, dest, symlinks=False)
        else:
            dest.mkdir(parents=True)
        shutil.copytree(log_dir, dest / "_runner_logs")
        for lf in swork.glob("exp_*.log"):
            shutil.copy(lf, dest / "_runner_logs" / lf.name)
        summary = summarize(dest)
        if summary.get("n_parts") and summary["n_part_meshes"] < summary["n_parts"]:
            warnings.append(f"only {summary['n_part_meshes']} part meshes for {summary['n_parts']} VLM parts "
                            "(3_split.py drops labels that receive no faces)")
        gpu_stages = [attempts[-1] if attempts else {}] + [stages.get("decoder", {})]
        meta = dict(base_meta, seed=seed, decoder_seed=seed,
                    wall_s=round(sum(s.get("wall_s", 0) for s in gpu_stages + [stages.get("split", {}), stages.get("simready", {})]), 1),
                    wall_s_invocation_this_seed=round(time.time() - t0, 1),
                    peak_gpu_mem_mib={"vlm_nvml": gpu_stages[0].get("nvml_peak_mib"),
                                      "vlm_torch_reserved": gpu_stages[0].get("torch_max_reserved_mib"),
                                      "decoder_nvml": gpu_stages[1].get("nvml_peak_mib"),
                                      "decoder_torch_reserved": gpu_stages[1].get("torch_max_reserved_mib")},
                    vlm_precision_used=attempts[-1].get("precision") if attempts else None,
                    stages=stages, summary=summary, warnings=warnings,
                    status="ok" if summary.get("has_basic.xml") and summary.get("has_basic.urdf") else "failed")
        (dest / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[physx-anything] seed {seed}: {meta['status']} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
