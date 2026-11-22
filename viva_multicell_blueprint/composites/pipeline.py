"""``blueprint_pipeline``: the single Registry-visible composite the Blueprint Studio
calls, assembled from the per-concern builders in :mod:`..compile.builders`.

* No subcellular layer: delegates to the classic ``compile()`` path, so the document
  is IDENTICAL to ``sego_crossparadigm`` (Rules writes the backend inputs directly).
* Subcellular layer present: Rules -> ``rules_*`` stores; SubcellularProcess reads the
  backend outputs and writes ``req_*``; CellActuationMergeStep reads both and is the
  SOLE writer of the backend's fates/divisions/deaths/target_size/motility/adhesion
  inputs (see :func:`..compile._shared._wire_subcellular`).
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from ..compile import builders
from ..compile import compile as _compile
from ..compile._shared import assemble
from ..core import register_workspace_processes
from .sego import _SEGO, _load_blueprint


@composite_generator(
    name="blueprint_pipeline",
    description=(
        "Backend-neutral blueprint compiled onto a mechanics backend (cpm|vertex|munk): "
        "fields/rules/ODE/immune mechanism, plus an optional per-cell subcellular layer "
        "arbitrated into the mechanics inputs by the actuation merge."
    ),
    parameters={
        "backend": {"type": "string", "default": "cpm",
                    "description": "mechanics substrate: 'cpm', 'vertex', or 'munk'"},
        "blueprint": {"type": "string", "default": str(_SEGO),
                      "description": "a path to a YAML file, inline YAML text, or a dict "
                                     "(v1 flat or v2 sectioned)"},
        "video_out": {"type": "string", "default": "",
                      "description": "if set, record a spatial-playback video to "
                                     "reports/assets/_studio_runs/<video_out>.html"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=80,
)
def blueprint_pipeline(core=None, backend: str = "cpm", blueprint=None,
                       video_out: str = "") -> dict:
    bp = _load_blueprint(blueprint)
    if not bp.subcellular:
        return _compile(bp, backend=backend, video_out=(video_out or None))
    return assemble(
        bp, builders.build_mechanics(bp, backend), video_out=(video_out or None),
        backend=backend, subcellular_nodes=builders.build_subcellular(bp),
        merge_node=builders.build_merge(bp),
    )
