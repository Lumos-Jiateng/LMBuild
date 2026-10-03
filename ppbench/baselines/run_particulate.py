#!/usr/bin/env python3
"""Particulate (RuiningLi/particulate, CVPR 2026): one static mesh -> parts + kinematic tree + joint limits.
Standalone; imports nothing from ppbench.

Mirrors infer.py of the official repo (Particulate-B, configs/particulate-B.yaml, PartField objaverse
features, num_points 102400 with 50% sharp-edge samples, min_part_confidence 0.0, single hypothesis).
Differences from infer.py, all recorded in meta.json:
  * weights come from the HF cache at pinned revisions;
  * meshes with more faces than `--max_faces` (default 50000) are quadric-decimated with pymeshlab first.
    The official HF demo rejects meshes with >51.2k faces ("please reduce the face count") and the
    README requires #uniform points (51.2k) > #faces; the decimated mesh is saved in native/;
  * `--strict auto` (default): strict connected-component refinement (README default) only when the input has
    >1 face-connected component and the largest covers <50% of faces; otherwise `--no_strict` (a single blob
    plus a few floaters is not "clean" components; strict collapses it to ~1-2 parts) (README: "If the input mesh
    does not have clean connected components, please specify --no_strict");
  * seeds fixed: torch.manual_seed(42) (as infer.py at import) and numpy.random.seed(0) (infer.py leaves
    numpy unseeded; point sampling uses numpy);
  * besides the official GLB/URDF exports, the raw prediction arrays and per-part meshes (in Particulate's
    +Z-up normalized frame) plus the input->model frame transform are saved to native/ so that
    ppbench/baselines/to_bundle_particulate.py can convert without re-running.

Run with Particulate's own venv from any cwd:
    CUDA_VISIBLE_DEVICES=4 flock /tmp/ppbench_gpu.lock \
      third_party/particulate/.venv/bin/python ppbench/baselines/run_particulate.py \
      --input MESH --system particulate-cube3d --seed 0 --out results/v2/<obj>/external \
      --up_dir Y --front +z
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("PARTICULATE_REPO", ROOT / "third_party/particulate"))
WEIGHTS = ("rayli/Particulate", "096167e661feb92a443535d15916323ec8a01613", "model.pt")
PARTFIELD = ("mikaelaangel/partfield-ckpt", "90b9b1e08b6a12fdcb6ee26b4854a26235e1765f", "model_objaverse.ckpt")


def git_rev(p):
    try:
        return subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def up_rotation(up_dir):
    """Exactly the matrices of infer.predict_mesh (vertices @ R.T)."""
    import numpy as np
    R = {"X": [[0, 0, -1], [0, 1, 0], [1, 0, 0]], "-X": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
         "Y": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "-Y": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
         "Z": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "-Z": [[1, 0, 0], [0, -1, 0], [0, 0, -1]]}[up_dir]
    return np.array(R, dtype=np.float64)


def load_mesh(path):
    """As infer.infer_single_mesh (glb/ply branch)."""
    import trimesh
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)


def cc_sizes(mesh):
    import numpy as np, trimesh
    cc = trimesh.graph.connected_components(edges=mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
    return sorted((len(c) for c in cc), reverse=True)


def n_components(mesh):
    return len(cc_sizes(mesh))


def estimate_front_yup(mesh):
    """Seat-facing direction of a +Y-up chair: the backrest (top 25% of height) sits on the back side."""
    import numpy as np
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    pts = mesh.sample(200000)
    top = pts[pts[:, 1] > lo[1] + 0.75 * (hi[1] - lo[1])]
    off = top.mean(0) - (lo + hi) / 2
    k = 0 if abs(off[0]) > abs(off[2]) else 2
    return ("-" if off[k] > 0 else "+") + "xyz"[k], off.round(4).tolist()


def decimate(mesh, target):
    import pymeshlab, trimesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    ms.meshing_merge_close_vertices()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), preservenormal=True,
                                                preservetopology=True, qualitythr=0.3, planarquadric=True)
    ms.meshing_remove_unreferenced_vertices()
    m = ms.current_mesh()
    out = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)
    return out, "pymeshlab meshing_merge_close_vertices + meshing_decimation_quadric_edge_collapse(" \
                f"targetfacenum={int(target)}, preservenormal, preservetopology, qualitythr=0.3, planarquadric)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--system", required=True, help="e.g. particulate-cube3d")
    ap.add_argument("--seed", type=int, required=True, help="seed of the upstream generated mesh (naming only)")
    ap.add_argument("--out", required=True, help="parent dir; run goes to <out>/<system>_s<seed>/")
    ap.add_argument("--up_dir", default="Y", choices=["X", "Y", "Z", "-X", "-Y", "-Z"])
    ap.add_argument("--front", required=True, help="signed axis the seat faces in the INPUT frame, e.g. +z, or auto")
    ap.add_argument("--num_points", type=int, default=102400)
    ap.add_argument("--min_part_confidence", type=float, default=0.0)
    ap.add_argument("--strict", default="auto", choices=["auto", "strict", "no_strict"])
    ap.add_argument("--max_faces", type=int, default=50000)
    ap.add_argument("--gpu_mem_cap_gb", type=float, default=0.0,
                    help="optional hard per-process cap via torch.cuda.set_per_process_memory_fraction (0 = none); default settings need ~48 GB")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    inp = Path(a.input).resolve()
    run = Path(a.out).resolve() / f"{a.system}_s{a.seed}"
    native = run / "native"
    (native / "parts").mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)  # infer.py / partfield_utils resolve configs and PartField/model relative to the repo
    sys.path.insert(0, str(REPO))

    import numpy as np, torch, trimesh
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    import infer as P  # official script as a module (sets torch.random.manual_seed(42) at import)
    from particulate.models import PAT_B  # noqa

    warnings = []
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    if a.gpu_mem_cap_gb:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_cap_gb / total_gb), 0)
    stage_peak = {}
    _orig_pf = P.obtain_partfield_feats

    def _pf_logged(*args, **kw):  # record the PartField stage peak separately (no numerical change)
        torch.cuda.reset_peak_memory_stats()
        out = _orig_pf(*args, **kw)
        torch.cuda.synchronize()
        stage_peak["partfield_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        torch.cuda.reset_peak_memory_stats()
        return out
    P.obtain_partfield_feats = _pf_logged
    t0 = time.time()
    cfg = OmegaConf.load(REPO / "configs/particulate-B.yaml")
    model_size = cfg.get("model_size", "B"); cfg.pop("model_size", None)
    model = eval(f"P.PAT_{model_size}")(**cfg).eval()
    ckpt = hf_hub_download(repo_id=WEIGHTS[0], filename=WEIGHTS[2], revision=WEIGHTS[1])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to("cuda")
    (REPO / "PartField/model").mkdir(parents=True, exist_ok=True)
    pf = hf_hub_download(repo_id=PARTFIELD[0], filename=PARTFIELD[2], revision=PARTFIELD[1])
    dst = REPO / "PartField/model/model_objaverse.ckpt"
    if not dst.exists():
        os.symlink(pf, dst)
    t_load = time.time() - t0

    mesh_in = load_mesh(inp)
    stats_in = {"n_vertices": int(len(mesh_in.vertices)), "n_faces": int(len(mesh_in.faces)),
                "n_components": n_components(mesh_in),
                "bbox_min": mesh_in.vertices.min(0).round(4).tolist(), "bbox_max": mesh_in.vertices.max(0).round(4).tolist()}
    mesh, decim = mesh_in, None
    if len(mesh_in.faces) > a.max_faces:
        mesh, how = decimate(mesh_in, a.max_faces)
        decim = {"method": how, "n_faces_before": int(len(mesh_in.faces)), "n_faces_after": int(len(mesh.faces)),
                 "n_components_after": n_components(mesh), "file": "native/input_decimated.ply"}
        mesh.export(native / "input_decimated.ply")
        warnings.append(f"input decimated {len(mesh_in.faces)} -> {len(mesh.faces)} faces (Particulate supports <51.2k faces)")
    sizes = cc_sizes(mesh)
    frac = sizes[0] / len(mesh.faces)
    strict = (len(sizes) > 1 and frac < 0.5) if a.strict == "auto" else (a.strict == "strict")
    if a.strict == "auto":
        warnings.append(f"strict={strict} chosen automatically: model input has {len(sizes)} face-connected component(s), "
                        f"largest covers {frac:.4f} of faces (rule: strict iff >1 component and largest < 50%)")
    front, front_evidence = a.front, None
    if a.front == "auto":
        assert a.up_dir == "Y", "--front auto assumes a +Y-up input"
        np.random.seed(0)
        front, off = estimate_front_yup(mesh_in)
        front_evidence = f"estimated: backrest (top 25% height) centroid offset from bbox center {off}; seat faces the other way"
        warnings.append(f"front {front} estimated automatically ({front_evidence})")

    torch.manual_seed(42); np.random.seed(0)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    outputs, face_indices, mesh_t = P.predict_mesh(mesh=mesh, up_dir=a.up_dir, model=model,
                                                  num_points=a.num_points, min_part_confidence=a.min_part_confidence)
    torch.cuda.synchronize()
    stage_peak["particulate_net_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    t_net = time.time() - t1
    (mesh_parts, face_part_ids, unique_part_ids, hierarchy, is_rev, is_pri,
     rev_plucker, rev_range, pri_axis, pri_range) = P.save_articulated_meshes(
        mesh_t, face_indices, outputs, output_path=native, strict=strict, animation_frames=50, save_name=None)
    P.export_urdf(mesh_parts, unique_part_ids, hierarchy, is_rev, is_pri, rev_plucker, rev_range, pri_axis,
                  pri_range, output_path=str(native / "urdf" / "model.urdf"), name="model")
    wall = time.time() - t0

    # recover the exact input->model transform used by predict_mesh: v_t = (v @ R.T - center) / scale
    R = up_rotation(a.up_dir)
    vr = mesh.vertices @ R.T
    lo, hi = vr.min(0), vr.max(0)
    center, scale = (lo + hi) / 2, float((hi - lo).max())
    assert np.allclose((vr - center) / scale, mesh_t.vertices, atol=1e-4), "frame transform mismatch"
    for pid, mp in zip(unique_part_ids, mesh_parts):
        mp.export(native / "parts" / f"part_{int(pid):02d}.ply")
    o = outputs[0]
    np.savez(native / "prediction.npz",
             unique_part_ids=np.asarray(unique_part_ids), face_part_ids=np.asarray(face_part_ids),
             motion_hierarchy=np.asarray(hierarchy, dtype=np.int64).reshape(-1, 2),
             is_part_revolute=is_rev, is_part_prismatic=is_pri, revolute_plucker=rev_plucker,
             revolute_range=rev_range, prismatic_axis=pri_axis, prismatic_range=pri_range,
             point_part_ids=np.asarray(o["part_ids"]),
             rotation=R, center=center, scale=scale)

    kinds = {}
    parents = {int(c): int(p) for p, c in hierarchy}
    for pid in unique_part_ids:
        pid = int(pid)
        k = ("revolute+prismatic" if is_rev[pid] and is_pri[pid] else "revolute" if is_rev[pid]
             else "prismatic" if is_pri[pid] else "fixed" if pid in parents else "root")
        kinds[k] = kinds.get(k, 0) + 1
    meta = {
        "system": a.system, "model": "Particulate-B",
        "code_repo": "https://github.com/RuiningLi/particulate", "code_commit": git_rev(REPO),
        "weights": {"repo": WEIGHTS[0], "revision": WEIGHTS[1], "file": WEIGHTS[2]},
        "partfield_weights": {"repo": PARTFIELD[0], "revision": PARTFIELD[1], "file": PARTFIELD[2]},
        "input_mesh": str(inp), "input_stats": stats_in, "input_front": front, "input_front_source": front_evidence or "declared (--front)",
        "seed": a.seed, "seed_meaning": "seed of the upstream generator run that produced the input mesh; "
                                        "Particulate itself runs with torch.manual_seed(42), numpy.random.seed(0)",
        "rotation_applied": {"up_dir_arg": a.up_dir,
                             "matrix_R": R.tolist(),
                             "formula": "v_model = (v_input @ R.T - center) / scale  (infer.predict_mesh: rotate to +Z up, "
                                        "center bbox, divide by max bbox extent)",
                             "center": center.round(6).tolist(), "scale": scale,
                             "reason": "Particulate is trained on +Z-up meshes (README); inputs are glTF +Y up"},
        "decimation": decim,
        "settings": {"model_config": "configs/particulate-B.yaml", "num_points": a.num_points,
                     "sharp_point_ratio": 0.5, "num_points_global_partfield": 40000,
                     "min_part_confidence": a.min_part_confidence, "strict": strict, "strict_arg": a.strict,
                     "hypothesis": 0, "animation_frames": 50},
        "wall_s": round(wall, 1), "wall_s_model_load": round(t_load, 1), "wall_s_inference": round(t_net, 1),
        "peak_gpu_mem_gb": {"torch_max_allocated_overall": max(stage_peak.values()),
                            "torch_max_reserved_overall": round(torch.cuda.max_memory_reserved() / 2**30, 2), **stage_peak,
                            "cap_gb": a.gpu_mem_cap_gb},
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
        "n_parts": int(len(unique_part_ids)), "motion_hierarchy": [[int(p), int(c)] for p, c in hierarchy],
        "part_motion_kinds": kinds,
        "native_files": sorted(str(p.relative_to(run)) for p in native.rglob("*") if p.is_file()),
        "output_convention": {"native_frame": "Particulate model frame: +Z up (input rotated by R), bbox-centered, "
                                              "max extent 1; plucker/axes/prismatic ranges in this frame; revolute "
                                              "ranges in radians",
                              "units": "unitless"},
        "notes": a.notes,
        "warnings": warnings,
    }
    meta["native_files"] = sorted(str(p.relative_to(run)) for p in native.rglob("*") if p.is_file())
    (run / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("system", "seed", "n_parts", "part_motion_kinds", "wall_s",
                                           "peak_gpu_mem_gb", "warnings")}), flush=True)


if __name__ == "__main__":
    main()
