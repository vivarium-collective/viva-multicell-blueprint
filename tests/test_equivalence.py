"""TOST equivalence metric + figure."""
import numpy as np
import plotly.graph_objects as go

from viva_multicell_blueprint.compare.metrics import equivalence_tost
from viva_multicell_blueprint.viz import equivalence_plot
from viva_multicell_blueprint.viz._common import HTML_CAP


def test_tost_declares_equivalence_when_cross_matches_same():
    rng = np.random.default_rng(0)
    same = 0.94 + rng.normal(0, 0.01, 30)
    cross = 0.935 + rng.normal(0, 0.01, 36)   # ~same, tiny loss
    r = equivalence_tost(same, cross, margin=0.05)
    assert r["equivalent"] is True
    assert r["p_tost"] < 0.05
    assert -0.05 < r["ci_low"] and r["ci_high"] < 0.05


def test_tost_rejects_equivalence_when_cross_much_worse():
    rng = np.random.default_rng(1)
    same = 0.94 + rng.normal(0, 0.01, 30)
    cross = 0.60 + rng.normal(0, 0.03, 36)    # large loss -> not equivalent
    r = equivalence_tost(same, cross, margin=0.05)
    assert r["equivalent"] is False
    assert r["diff"] > 0.05                    # cross agrees much less


def test_tost_handles_degenerate_small_samples():
    r = equivalence_tost([0.9], [0.9, 0.9], margin=0.05)
    assert r["equivalent"] is False and "reason" in r


def test_equivalence_plot_renders_under_cap():
    rng = np.random.default_rng(2)
    same = 0.94 + rng.normal(0, 0.01, 30)
    cross = 0.935 + rng.normal(0, 0.01, 36)
    tost = equivalence_tost(same, cross, margin=0.05)
    fig = equivalence_plot(same, cross, tost)
    assert isinstance(fig, go.Figure)
    boxes = [t for t in fig.data if t.type == "box"]
    assert len(boxes) == 2
    assert len(fig.to_html(include_plotlyjs="cdn").encode()) < HTML_CAP
