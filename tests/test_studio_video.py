"""Per-run spatial video: single_playback + the SpatialVideoStep recorder + the
compile wiring that adds it when `video_out` is set."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compare.runner import run_backend
from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.composites.sego import sego_crossparadigm
from viva_multicell_blueprint.viz import single_playback

ROOT = Path(__file__).resolve().parents[1]
_BP = {
    "domain": {"extent_um": [16, 16], "grid_resolution_um": 1, "boundary": "noflux"},
    "cell_types": [{"name": "H", "target_size": 25}],
    "initial_state": {"kind": "blocks",
                      "block": {"type": "H", "origin": [4, 4], "shape": [2, 2], "spacing": 5, "size": 25}},
}


def test_single_playback_renders():
    run = run_backend(Blueprint.from_dict(dict(_BP)), "cpm", 4)
    fig = single_playback(run.frames, {"extent_um": [16, 16], "grid_resolution_um": 1}, backend="cpm")
    assert isinstance(fig, go.Figure) and fig.frames


def test_recorder_is_registered():
    assert "SpatialVideoStep" in build_core().link_registry


def test_compile_adds_video_node_only_when_requested():
    without = sego_crossparadigm(backend="cpm", blueprint=_BP)
    assert "spatial_video" not in without["state"]
    withv = sego_crossparadigm(backend="cpm", blueprint=_BP, video_out="../../etc/evil name")
    node = withv["state"].get("spatial_video")
    assert node and node["address"] == "local:SpatialVideoStep"
    # a path-traversal attempt is sanitized to a flat basename that resolves INSIDE
    # the _studio_runs dir (no separators survive, so no escape)
    out = Path(node["config"]["out_path"]).resolve()
    runs = (ROOT / "workspace" / "reports" / "assets" / "_studio_runs").resolve()
    assert out.parent == runs and "/" not in node["config"]["out_path"].rsplit("_studio_runs/", 1)[-1]


def test_run_writes_a_video_file(tmp_path):
    core = build_core()
    doc = sego_crossparadigm(core=core, backend="cpm", blueprint=_BP, video_out="pytest_vid")
    from process_bigraph import Composite
    Composite(doc, core=core).run(3)
    out = ROOT / "workspace" / "reports" / "assets" / "_studio_runs" / "pytest_vid.html"
    assert out.is_file() and "Plotly.newPlot" in out.read_text()[:300000]
    out.unlink()  # don't leave the test artifact
