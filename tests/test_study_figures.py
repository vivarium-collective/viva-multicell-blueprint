"""Unit tests for the per-study summary figure builders (viz.study_figures).

These are pure functions over already-computed arrays, so the tests feed small
synthetic data and assert the figure shape + that each renders well under the
inline HTML cap the report applies.
"""
import plotly.graph_objects as go

from viva_multicell_blueprint.viz import adhesion_sweep, dose_response, t0_fidelity
from viva_multicell_blueprint.viz._common import HTML_CAP


def _under_cap(fig):
    return len(fig.to_html(include_plotlyjs="cdn").encode()) < HTML_CAP


def test_adhesion_sweep_two_panels_and_cpm_reference():
    scales = [0.5, 1.0, 2.0, 4.0]
    fig = adhesion_sweep(scales, vertex_mean=[24, 25, 26, 27], vertex_cv=[0.1, 0.11, 0.12, 0.13],
                         cpm_mean=25.0, cpm_cv=0.1, target_size=25)
    assert isinstance(fig, go.Figure)
    # one vertex trace per panel = 2 vertex series; exactly one legend entry
    vtx = [t for t in fig.data if t.name == "vertex"]
    assert len(vtx) == 2 and sum(1 for t in vtx if t.showlegend) == 1
    assert _under_cap(fig)


def test_dose_response_direction_and_one_legend_per_backend():
    doses = [0.03, 0.08, 0.15]
    series = [
        {"backend": "cpm", "seed": 0, "final_I": [2, 5, 9]},
        {"backend": "cpm", "seed": 1, "final_I": [1, 4, 8]},
        {"backend": "vertex", "seed": 0, "final_I": [2, 6, 10]},
        {"backend": "vertex", "seed": 1, "final_I": [3, 5, 11]},
    ]
    fig = dose_response(doses, series)
    for backend in ("cpm", "vertex"):
        assert sum(1 for t in fig.data if t.name == backend and t.showlegend) == 1
    # the backend series carry the real data (4 of them)
    data_traces = [t for t in fig.data if t.name in ("cpm", "vertex")]
    assert len(data_traces) == 4
    assert _under_cap(fig)


def test_t0_fidelity_box_per_backend_with_rms_in_name():
    fig = t0_fidelity({"cpm": [0.0, 0.0, 1e-7], "vertex": [0.3, 1.1, 0.8]},
                      type_ok={"cpm": 48, "vertex": 48}, n_cells=48)
    boxes = [t for t in fig.data if t.type == "box"]
    assert len(boxes) == 2
    assert any("RMS" in (t.name or "") for t in boxes)
    assert _under_cap(fig)
