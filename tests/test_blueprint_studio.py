"""Blueprint Studio: inline-blueprint generator + the Explore-tab viewer descriptor."""
from __future__ import annotations

from pathlib import Path

import yaml

from viva_multicell_blueprint import workbench_viewers
from viva_multicell_blueprint.composites.sego import _load_blueprint, sego_crossparadigm

ROOT = Path(__file__).resolve().parents[1]

_INLINE = """
blueprint_version: "1.0"
domain: {extent_um: [20, 20], grid_resolution_um: 1, boundary: noflux}
cell_types: [{name: H, target_size: 25}]
initial_state: {kind: blocks, block: {type: H, origin: [5, 5], shape: [2, 2], spacing: 5, size: 25}}
"""


def test_load_blueprint_accepts_inline_yaml_dict_and_path():
    bp_yaml = _load_blueprint(_INLINE)
    bp_dict = _load_blueprint(yaml.safe_load(_INLINE))
    assert bp_yaml.to_dict() == bp_dict.to_dict()
    # path still works
    path = ROOT / "workspace" / "blueprints" / "examples" / "simple_sheet.yaml"
    bp_path = _load_blueprint(str(path))
    assert bp_path.cell_types[0]["name"] == "H"
    # None -> default Sego
    assert _load_blueprint(None).cell_types


def test_generator_runs_inline_blueprint_on_both_backends():
    for backend in ("cpm", "vertex"):
        doc = sego_crossparadigm(backend=backend, blueprint=_INLINE)
        assert "state" in doc


def test_workbench_viewers_registers_blueprint_studio():
    viewers = workbench_viewers.get_viewers(ROOT)
    assert any(v["id"] == "blueprint-studio" and v["kind"] == "launcher" for v in viewers)
    v = next(v for v in viewers if v["id"] == "blueprint-studio")
    out = v["launch"](ROOT, None, None, {"public_base": "https://x.test/wb"})
    assert out["url"] == "https://x.test/wb/assets/blueprint_studio.html"
    # root-relative fallback when no public_base
    assert v["launch"](ROOT, None, None, {})["url"] == "/assets/blueprint_studio.html"


def test_studio_app_is_served_file():
    assert (ROOT / "workspace" / "reports" / "assets" / "blueprint_studio.html").is_file()


def test_registered_generator_builds_with_core_kwarg():
    # The framework calls the generator as fn(core=..., **params); a decorator bound
    # to the wrong function (the earlier bug) raised "unexpected keyword 'core'".
    import viva_multicell_blueprint.composites  # noqa: F401 — register
    from process_bigraph.composite_generator import _cs, build_generator
    sid = "viva_multicell_blueprint.composites.sego.sego_crossparadigm"
    spec = _cs.get(sid)
    assert spec is not None, "generator not registered under the expected id"
    doc = build_generator(spec, overrides={"backend": "cpm", "blueprint": _INLINE})
    assert isinstance(doc, dict)


def test_served_sego_copy_matches_source():
    """The Studio serves a copy of the Sego blueprint at reports/assets/ (the
    blueprints dir isn't web-served). Guard against it drifting from the source."""
    src = yaml.safe_load((ROOT / "workspace" / "blueprints" / "influenza_sego2022.yaml").read_text())
    served = yaml.safe_load((ROOT / "workspace" / "reports" / "assets" / "sego_blueprint.yaml").read_text())
    assert served == src, "reports/assets/sego_blueprint.yaml drifted from the source blueprint; re-copy it"
