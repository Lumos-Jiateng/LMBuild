"""LMBuild v2: articulated mesh assemblies against the benchmark_v2 tasks.

The v1 package scores brick designs on the LDraw stud lattice. Benchmark v2
tasks give a mixed pool of metric part meshes (Artiverse, PartNeXt, Fusion 360,
standard hardware), cited real-world claims and a reference object with
annotated joints, so the medium changes: a design is a set of posed part
meshes in metres with roles, materials, joints and an assembly sequence.

    task.py        load a core_v2 task, its pool and its reference (Z-up, metres, ground at z=0)
    glb.py         a dependency-light GLB reader (node tree, extras, transforms)
    design.py      the design record every producer emits, saved as manifest + GLB
    csg.py         Tier B part creation: a JSON CSG program compiled by manifold3d
    materials.py   the material vocabulary and densities, taken from Artiverse annotations
    lexicon.py     free-text part names -> canonical roles, per object type
    voxel.py       the one geometric primitive: filled voxel occupancy per part
    env.py         the tool protocol (Tier A catalog, Tier B catalog + creation)
    session.py     shell transport for terminal agents
    loop.py        OpenAI-compatible transport with text tool calls, for local VLMs
    render.py      Blender (Cycles, CPU) renders of a design
    evaluate.py    the dimensions of docs/evaluation, graded, with depth labels
    population.py  category statistics from the Artiverse population (priors)

Everything runs in `.venv_eval` (trimesh, manifold3d, scipy); rendering crosses
into Blender's Python in a subprocess.
"""
