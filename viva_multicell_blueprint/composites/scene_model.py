"""``scene_model``: a registered composite that runs a model-definition-format scene.

Loads a scene YAML plus a per-type model registry and hands both to
:func:`..model_format.compile.compile_scene`. The default demo mixes three
formalisms (ODE, Smoldyn particles, MaBoSS Boolean) sharing one extracellular field.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from ..core import register_workspace_processes
from ..model_format.compile import SceneModelProcess  # noqa: F401 (registers it)
from ..model_format.compile import compile_scene
from ..model_format.registry import ModelSpec, Registry, Transport
from ..model_format.scene import Scene

_SCENE = Path(__file__).resolve().parents[2] / "workspace" / "scenes" / "mixed_formalism_demo.yaml"

_NET = {
    "species": {"A": 4.0, "B": 4.0, "C": 0.0},
    "reactions": [{"subs": ["A", "B"], "prds": ["C"], "kf": 0.05, "kb": 0.5}],
}
_BNET = {"X": "!Y", "Y": "X"}


def demo_registry() -> Registry:
    """metabolic -> ODE, signalling -> particle, fate -> Boolean (MaBoSS)."""
    return Registry({
        "metabolic": ModelSpec(formalism="ode", source=dict(_NET),
                               membrane={"C": Transport("cross", 0.1)}),
        "signalling": ModelSpec(formalism="particle", source=dict(_NET)),
        "fate": ModelSpec(formalism="maboss", source=_BNET),
    })


@composite_generator(
    name="scene_model",
    description=(
        "A scene of cells, each bound by type to a model in its own formalism "
        "(SBML-ODE, Smoldyn particles, MaBoSS Boolean) sharing one extracellular field."
    ),
    parameters={
        "scene": {"type": "string", "default": str(_SCENE),
                  "description": "path to a scene YAML"},
        "mechanics": {"type": "string", "default": "vertex",
                      "description": "tissue mechanics backend: 'vertex', 'cpm' or 'munk'"},
        "overlay_out": {"type": "string", "default": "scene_model",
                        "description": "reserved: overlay name under reports/assets/"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=12,
)
def scene_model(core=None, scene: str = str(_SCENE), mechanics: str = "vertex",
                overlay_out: str = "scene_model") -> dict:
    sc = Scene.from_yaml(Path(scene).read_text(encoding="utf-8"))
    return compile_scene(sc, demo_registry(), mechanics=str(mechanics))
