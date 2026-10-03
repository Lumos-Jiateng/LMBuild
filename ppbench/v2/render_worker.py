"""Blender worker for v2 renders. Runs inside Blender's Python, one job per process.

    <bpy python> render_worker.py -- job.json

job = {"mode": "design", "glb": path, "out": prefix, "views": {"iso": [az, el], ...},
       "colors": {node_name: [r, g, b]}, "size": 512, "samples": 24, "ground": true}
job = {"mode": "thumbs", "items": [{"glb": path, "out": png}], "view": [az, el], "size": 256, "samples": 16}

Cycles on the CPU only: EEVEE opens a GL context on whichever GPU EGL picks, and
on this machine that is someone else's. The glTF importer turns +Y up into +Z
up with (x, y, z) -> (x, -z, y), the same map ppbench.v2.task applies, so a
design GLB written in the world frame is converted back by writing it Y-up
first (render.py does that) and a pool GLB can be imported as it is.
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
from mathutils import Vector


def reset(size, samples, view_transform="AgX", world_strength=0.9, light_scale=1.0):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.render.threads_mode = "FIXED"
    sc.render.threads = int(os.environ.get("RENDER_THREADS", "16"))
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    for vt in (view_transform, "AgX", "Filmic", "Standard"):
        try:
            sc.view_settings.view_transform = vt
            break
        except Exception:
            continue
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.93, 0.93, 0.95, 1)
    bg.inputs[1].default_value = world_strength
    sc.world = world
    for name, rot, energy in (("key", (50, 10, 35), 3.0), ("fill", (70, -20, -120), 1.0), ("rim", (110, 0, 180), 0.8)):
        s = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        s.data.energy = energy * light_scale
        s.rotation_euler = tuple(math.radians(a) for a in rot)
        bpy.context.collection.objects.link(s)
    return sc


def material(name, rgb, rough=0.55):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1)
    b.inputs["Roughness"].default_value = rough
    return m


def meshes():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def bounds(objs):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for ob in objs:
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            lo = Vector((min(lo[i], w[i]) for i in range(3)))
            hi = Vector((max(hi[i], w[i]) for i in range(3)))
    return lo, hi


def camera(sc, lo, hi):
    c = (lo + hi) / 2
    r = max((hi - lo).length / 2, 1e-4)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.collection.objects.link(cam)
    sc.camera = cam
    cam.data.lens = 50
    cam.data.clip_start = r / 200
    cam.data.clip_end = r * 200
    fov = 2 * math.atan(cam.data.sensor_width / (2 * cam.data.lens))
    return cam, c, r / math.sin(fov / 2) * 1.05


def shoot(sc, cam, c, dist, az, el, path):
    a, e = math.radians(az), math.radians(el)
    loc = c + Vector((dist * math.cos(e) * math.cos(a), dist * math.cos(e) * math.sin(a), dist * math.sin(e)))
    cam.location = loc
    cam.rotation_euler = (c - loc).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def design(job):
    sc = reset(job.get("size", 512), job.get("samples", 24), job.get("view_transform", "AgX"), job.get("world_strength", 0.9),
               job.get("light_scale", 1.0))
    bpy.ops.import_scene.gltf(filepath=job["glb"])
    objs = meshes()
    colors = job.get("colors", {})
    for ob in objs:
        rgb = colors.get(ob.name) or colors.get(ob.data.name) or [0.6, 0.62, 0.66]
        ob.data.materials.clear()
        ob.data.materials.append(material("m_" + ob.name, rgb))
    lo, hi = bounds(objs)
    if job.get("frame"):          # optional framing box (world, Z-up), e.g. to zoom on a small moving part
        lo, hi = Vector(job["frame"][0]), Vector(job["frame"][1])
    if job.get("ground", True):
        span = max((hi - lo).length, 0.5) * 3
        bpy.ops.mesh.primitive_plane_add(size=span, location=((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, 0.0))
        pl = bpy.context.active_object
        pl.data.materials.append(material("ground", job.get("ground_rgb", [0.82, 0.82, 0.8]), 0.9))
    cam, c, dist = camera(sc, lo, hi)
    for name, (az, el) in job["views"].items():
        shoot(sc, cam, c, dist, az, el, f"{job['out']}_{name}.png")
    if job.get("id_colors"):      # optional flat id pass from the same cameras: <out>_<view>_id.png, emission colours through the sRGB transform
        id_pass(sc, objs, job["id_colors"])
        for name, (az, el) in job["views"].items():
            shoot(sc, cam, c, dist, az, el, f"{job['out']}_{name}_id.png")


def id_pass(sc, objs, id_colors):
    keep = {ob.name for ob in objs}
    for ob in list(sc.objects):
        if ob.name not in keep and ob.type in ("LIGHT", "MESH"):
            bpy.data.objects.remove(ob, do_unlink=True)
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    sc.cycles.use_denoising = False
    for vt in ("Standard", "Raw"):
        try:
            sc.view_settings.view_transform = vt
            break
        except Exception:
            continue
    sc.view_settings.look = "None"
    for ob in objs:
        rgb = id_colors.get(ob.name) or id_colors.get(ob.data.name) or [0.0, 0.0, 0.0]
        m = bpy.data.materials.new("id_" + ob.name)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs[0].default_value = (rgb[0], rgb[1], rgb[2], 1)
        em.inputs[1].default_value = 1.0
        outn = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(em.outputs[0], outn.inputs[0])
        ob.data.materials.clear()
        ob.data.materials.append(m)


def thumbs(job):
    az, el = job.get("view", [-50, 28])
    for it in job["items"]:
        sc = reset(job.get("size", 256), job.get("samples", 16))
        bpy.ops.import_scene.gltf(filepath=it["glb"])
        objs = meshes()
        for ob in objs:
            ob.data.materials.clear()
            ob.data.materials.append(material("clay", [0.45, 0.55, 0.72]))
        lo, hi = bounds(objs)
        cam, c, dist = camera(sc, lo, hi)
        shoot(sc, cam, c, dist, az, el, it["out"])


def main():
    job = json.loads(open(sys.argv[sys.argv.index("--") + 1]).read())
    {"design": design, "thumbs": thumbs}[job["mode"]](job)


main()
