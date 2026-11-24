"""Generate blueprint(s) initialized from real MIBI-TOF spatial-omics imaging.

Reads the committed per-cell table `data/spatial/mibitof_colon_cells.json` (a
MIBI-TOF human colon-carcinoma field of view — squidpy's `mibitof`, the same
modality as HuBMAP CODEX — reduced to per-cell {type, centroid, size} by
viva-cpm's segmentation pipeline; see that file's `provenance`) and writes two
blueprints whose `initial_state.cells` is seeded at the cells' real imaged
positions:

  * workspace/blueprints/mibitof_colon.yaml        — the full FOV (~1016 cells)
  * workspace/blueprints/mibitof_colon_patch.yaml  — a cropped dense patch
                                                     (~100 cells; Studio-scale)

No heavy deps: the imaging was reduced to the committed JSON once, so this is a
pure JSON->YAML transform. Re-run after changing the crop/scaling:

    python scripts/gen_mibitof_blueprint.py

Units follow the repo convention (`influenza_sego2022.yaml`): 1 grid unit = 1
lattice site; `*_um` keys are in lattice sites. The MIBI-TOF lattice is the 4x-
downsampled segmentation mask, so 1 site ~ 3.2 um of real tissue.
"""
from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "data", "spatial", "mibitof_colon_cells.json")
OUT_DIR = os.path.join(HERE, "workspace", "blueprints")
# The Studio fetches templates from reports/assets/ (alongside sego_blueprint.yaml),
# so write a serving copy there too. Both are regenerated each run (no drift).
ASSETS_DIR = os.path.join(HERE, "workspace", "reports", "assets")

# Structure-preserving adhesion (mirrors viva-cpm run_hra_mibitof.py): medium is
# costly so cells stay packed as imaged; cell-cell contacts are uniform.
J_MEDIUM = 16
J_CELL = 6
PATCH_W = 64          # crop window (lattice sites); ~100 cells at this density


def _yaml_cells(cells):
    lines = []
    for c in cells:
        cx, cy = c["centroid"]
        lines.append(f"    - {{id: {c['id']}, type: {c['type']}, "
                     f"centroid: [{cx}, {cy}], size: {c['size']}}}")
    return "\n".join(lines)


def _blueprint_text(name, note, cells, extent, type_names, target_size, prov):
    counts = Counter(c["type"] for c in cells)
    ct = "\n".join(f"  - {{name: {t}, target_size: {target_size}}}" for t in type_names)
    adh = []
    for t in type_names:
        adh.append(f"  - {{a: {t}, b: medium, strength: {J_MEDIUM}}}")
    for i, a in enumerate(type_names):
        for b in type_names[i:]:
            adh.append(f"  - {{a: {a}, b: {b}, strength: {J_CELL}}}")
    comp = ", ".join(f"{t} {counts[t]}" for t in type_names if counts[t])
    return f"""# {name}: a multicell blueprint initialized from REAL MIBI-TOF spatial-omics imaging.
# {note}
# Provenance: {prov['source']}
#   {prov['pipeline']} (FOV {prov['fov']}).
# Units: 1 grid unit = {prov['unit']}; `*_um` keys are in lattice sites.
# Composition ({len(cells)} cells): {comp}
blueprint_version: "1.0"
domain:
  extent_um: [{extent[0]}, {extent[1]}]
  grid_resolution_um: 1
  boundary: noflux

cell_types:
{ct}

# Structure-preserving contact energies (medium costly -> cells stay packed as
# imaged; uniform cell-cell), after viva-cpm run_hra_mibitof.py.
adhesion:
{chr(10).join(adh)}

initial_state:
  # Seeded at each cell's real imaged centroid + segmented size (no synthetic layout).
  cells:
{_yaml_cells(cells)}
"""


def main():
    doc = json.load(open(SRC))
    prov = doc["provenance"]
    cells = doc["cells"]
    type_names = list(prov["type_names"])
    target = int(prov["median_cell_size_px"])
    nx, ny = prov["lattice"]

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    def _emit(fname, text):
        for d in (OUT_DIR, ASSETS_DIR):
            with open(os.path.join(d, fname), "w") as fh:
                fh.write(text)

    # --- full FOV ---
    _emit("mibitof_colon.yaml", _blueprint_text(
        "mibitof_colon", "Full imaged field of view.",
        cells, [nx, ny], type_names, target, prov))
    print(f"wrote mibitof_colon.yaml  ({len(cells)} cells, {nx}x{ny})")

    # --- cropped dense patch: slide a PATCH_W window to the densest region ---
    best = None
    for y0 in range(0, ny - PATCH_W + 1, 8):
        for x0 in range(0, nx - PATCH_W + 1, 8):
            n = sum(1 for c in cells
                    if x0 <= c["centroid"][0] < x0 + PATCH_W
                    and y0 <= c["centroid"][1] < y0 + PATCH_W)
            if best is None or n > best[0]:
                best = (n, x0, y0)
    _, x0, y0 = best
    patch = [{"id": c["id"], "type": c["type"], "size": c["size"],
              "centroid": [round(c["centroid"][0] - x0, 2),
                           round(c["centroid"][1] - y0, 2)]}
             for c in cells
             if x0 <= c["centroid"][0] < x0 + PATCH_W
             and y0 <= c["centroid"][1] < y0 + PATCH_W]
    patch_types = [t for t in type_names if any(c["type"] == t for c in patch)]
    _emit("mibitof_colon_patch.yaml", _blueprint_text(
        "mibitof_colon_patch",
        f"Densest {PATCH_W}x{PATCH_W} crop of the FOV (origin [{x0},{y0}]).",
        patch, [PATCH_W, PATCH_W], patch_types, target, prov))
    print(f"wrote mibitof_colon_patch.yaml  ({len(patch)} cells, {PATCH_W}x{PATCH_W}, crop @ [{x0},{y0}])")


if __name__ == "__main__":
    main()
