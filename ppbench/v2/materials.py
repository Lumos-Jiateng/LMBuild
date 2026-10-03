"""The material vocabulary a design may declare, with densities from data.

The house rule is no invented numbers. The vocabulary and every density below
are the Artiverse annotations themselves: all `material.json` files under
`reference_data/.cache/artiverse_hub/data` (5,510 objects, surveyed
2026-09-13), grouped by (material, submaterial), density = median of the
annotated values, n = number of annotated parts. Nine classes, 24 submaterials.

Elastic moduli are not used by any metric yet and are therefore not listed.
"""
from __future__ import annotations

# submaterial id: (class, median density kg/m^3, n annotated parts)
LIBRARY = {
    "aluminum": ("light_metal", 2708.1, 8196),
    "stainless_steel": ("heavy_metal", 7954.1, 3166),
    "steel": ("heavy_metal", 7939.9, 2957),
    "polyvinyl_chloride_PVC": ("plastic", 1501.3, 7609),
    "polyethylene_PE": ("plastic", 948.0, 2821),
    "acrylic_PMMA": ("plastic", 1187.2, 2802),
    "ABS": ("plastic", 1054.7, 2667),
    "polypropylene_PP": ("plastic", 904.6, 2556),
    "particle_board_MDF": ("wood", 700.9, 3360),
    "plywood": ("wood", 549.8, 3295),
    "solid_wood_hardwood": ("wood", 650.1, 3234),
    "solid_wood_softwood": ("wood", 399.6, 3046),
    "borosilicate_glass": ("glass", 2242.1, 814),
    "tempered_glass": ("glass", 2519.7, 770),
    "soda_lime_glass": ("glass", 2495.1, 734),
    "porcelain": ("ceramic", 2397.5, 307),
    "silicone_rubber": ("rubber", 1083.6, 289),
    "synthetic_rubber": ("rubber", 1127.6, 266),
    "natural_rubber": ("rubber", 909.7, 244),
    "memory_foam": ("foam", 59.8, 158),
    "polyurethane_foam": ("foam", 40.3, 109),
    "cotton_fabric": ("textile", 246.6, 35),
    "polyester_mesh": ("textile", 312.1, 22),
}
SOURCE = "Artiverse material.json annotations, median per submaterial over 5,510 objects (surveyed 2026-09-13)"

CLASSES = sorted({c for c, _, _ in LIBRARY.values()})
TRANSPARENT = {"glass"}
COMPLIANT = {"rubber", "foam", "textile"}

# Words used in claims -> material classes. "metal" covers both metal classes.
CLAIM_WORDS = {
    "metal": {"heavy_metal", "light_metal"}, "steel": {"heavy_metal"}, "iron": {"heavy_metal"},
    "aluminum": {"light_metal"}, "aluminium": {"light_metal"}, "plastic": {"plastic"},
    "wood": {"wood"}, "wooden": {"wood"}, "timber": {"wood"}, "glass": {"glass"}, "rubber": {"rubber"},
    "foam": {"foam"}, "fabric": {"textile"}, "textile": {"textile"}, "cloth": {"textile"},
    "leather": {"textile"}, "mesh": {"textile"}, "ceramic": {"ceramic"}, "porcelain": {"ceramic"},
}
# Artiverse's own class ids as they appear in material.json
ARTIVERSE_CLASS = {c: c for c in CLASSES}


def resolve(name: str | None):
    """(submaterial, class, density) or None. Accepts a submaterial id, case-insensitive."""
    if not name:
        return None
    key = str(name).strip()
    for sub, (cls, dens, _) in LIBRARY.items():
        if sub.lower() == key.lower():
            return sub, cls, dens
    return None


def claim_classes(statement: str) -> list[set[str]]:
    """One group per material word in the claim; a group is satisfied by any class in it
    ("metal" is one requirement that heavy or light metal meets)."""
    import re
    out = []
    for w in re.findall(r"[a-z]+", statement.lower()):
        g = CLAIM_WORDS.get(w)
        if g and g not in out:
            out.append(set(g))
    return out


def listing() -> list[dict]:
    return [{"material": sub, "class": cls, "density_kg_m3": dens}
            for sub, (cls, dens, _) in sorted(LIBRARY.items(), key=lambda kv: (kv[1][0], kv[0]))]
