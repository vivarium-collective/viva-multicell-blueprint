"""Workspace composite generator (spec §5/§9): the Sego blueprint compiled onto a
mechanics backend, exposed to the dashboard's Composite Explorer and used as the
baseline composite of the ``influenza-crossparadigm`` studies.

One generator, parameterized by ``backend`` (``cpm``|``vertex``) and the
``blueprint`` path, so the same runnable artifact serves the full flu-lung study
and the mechanics-only adhesion-calibration sheet. It simply ``compile()``s the
backend-neutral blueprint -- the single source of truth -- so the Explorer shows
the real shared mechanism + the one backend-specific mechanics node.

``sego_crossparadigm_large`` is a thin preset wrapper over the same generator whose
``blueprint`` defaults to a biology-preserving ~2.4x scale-up (a bigger tissue, the
same mechanism); registering it the same way surfaces it as its own startable
simulation in the Blueprint Studio catalog without duplicating any mechanism.

Discovery: this lives in ``<pkg>/composites/*.py`` (a package directory, not a
single ``composites.py`` module) so the workbench's workspace-scope glob
(``_discover_workspace_generators``) imports it, and the top-level
``viva_multicell_blueprint/__init__.py`` does ``from . import composites`` so the
env-worker's ``import <package_path>`` registers it for the live server.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from ..blueprint.model import Blueprint
from ..compile import compile as _compile
from ..core import register_workspace_processes

# __file__ is <pkg>/composites/sego.py, so the repo root (which holds
# workspace/blueprints/) is three parents up.
_BLUEPRINTS = Path(__file__).resolve().parent.parent.parent / "workspace" / "blueprints"
_SEGO = _BLUEPRINTS / "influenza_sego2022.yaml"
# Biology-preserving ~2.4x scale-up of _SEGO (120x120 patch, 15x15 ~225-cell block).
# Same mechanism, bigger tissue -- see the blueprint header for the scale-up contract.
_SEGO_LARGE = _BLUEPRINTS / "influenza_sego2022_large.yaml"


def _load_blueprint(blueprint):
    """Resolve the ``blueprint`` parameter to a :class:`Blueprint`.

    Accepts (a) None -> the default Sego blueprint, (b) a dict -> loaded directly,
    (c) a path to an existing YAML file, or (d) inline YAML text. (c)/(d) let the
    Blueprint Studio UI run an edited blueprint without writing a file first.
    """
    import yaml
    if blueprint is None:
        return Blueprint.load(_SEGO)
    if isinstance(blueprint, dict):
        return Blueprint.from_dict(blueprint)
    text = str(blueprint)
    p = Path(text)
    if len(text) < 512 and p.exists():         # a path (guard against huge inline text)
        return Blueprint.load(p)
    doc = yaml.safe_load(text)                  # inline YAML
    if not isinstance(doc, dict):
        raise ValueError("blueprint must be a path, inline YAML mapping, or dict")
    return Blueprint.from_dict(doc)


@composite_generator(
    name="sego_crossparadigm",
    description=(
        "Sego 2022 influenza blueprint compiled onto a mechanics backend "
        "(cpm|vertex|munk): the shared field/rules/ODE/recruitment mechanism plus "
        "the one backend-specific mechanics node."
    ),
    parameters={
        "backend": {"type": "string", "default": "cpm",
                    "description": "mechanics substrate: 'cpm' (Cellular Potts), "
                                   "'vertex' (tyssue), or 'munk' (center-based pymunk)"},
        "blueprint": {"type": "string", "default": str(_SEGO),
                      "description": "backend-neutral blueprint: a path to a YAML file, "
                                     "inline YAML text, or a dict (Blueprint Studio sends inline YAML)"},
        "video_out": {"type": "string", "default": "",
                      "description": "if set, record a spatial-playback video to "
                                     "reports/assets/_studio_runs/<video_out>.html (Blueprint Studio)"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=80,
)
def sego_crossparadigm(core=None, backend: str = "cpm", blueprint: str | None = None,
                       video_out: str = "") -> dict:
    bp = _load_blueprint(blueprint)
    return _compile(bp, backend=backend, video_out=(video_out or None))   # {"state": {...}} envelope


@composite_generator(
    name="sego_crossparadigm_large",
    description=(
        "Scaled-up Sego 2022 influenza preset: a biology-preserving ~2.4x linear "
        "scale-up of 'sego_crossparadigm' (120x120 patch, 15x15 ~225-cell epithelial "
        "block) -- a bigger tissue, the SAME field/rules/ODE/recruitment mechanism. "
        "A heavier starting point to run from in the Blueprint Studio."
    ),
    parameters={
        "backend": {"type": "string", "default": "cpm",
                    "description": "mechanics substrate: 'cpm' (Cellular Potts), "
                                   "'vertex' (tyssue), or 'munk' (center-based pymunk)"},
        "blueprint": {"type": "string", "default": str(_SEGO_LARGE),
                      "description": "backend-neutral blueprint: a path to a YAML file, "
                                     "inline YAML text, or a dict (defaults to the scaled-up Sego preset)"},
        "video_out": {"type": "string", "default": "",
                      "description": "if set, record a spatial-playback video to "
                                     "reports/assets/_studio_runs/<video_out>.html (Blueprint Studio)"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=80,
)
def sego_crossparadigm_large(core=None, backend: str = "cpm", blueprint: str | None = None,
                             video_out: str = "") -> dict:
    """Thin preset wrapper over :func:`sego_crossparadigm`: identical generator, but
    its ``blueprint`` defaults to the larger scaled-up Sego blueprint so it surfaces
    as its own startable simulation in the Studio catalog. DRY -- no mechanism is
    duplicated; only the default starting geometry differs."""
    return sego_crossparadigm(core=core, backend=backend,
                              blueprint=(blueprint if blueprint is not None else str(_SEGO_LARGE)),
                              video_out=video_out)
