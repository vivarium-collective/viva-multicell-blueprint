"""Fast integration guard for the large-crossparadigm-comparison study: the large
blueprint is really large, the behavior metrics integrate with the 3-backend runner,
and center-based (munk) MOVES on the big tissue (the #24 fix, validated at scale).

Kept to tiny step counts (the 254-cell tissue is slow) — the full characterization is
scripts/compare_large_crossparadigm.py, not CI."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.compare.runner import run_backend
from viva_multicell_blueprint.compile._shared import resolve_initial_state

ROOT = Path(__file__).resolve().parents[1]
LARGE = ROOT / "workspace" / "blueprints" / "influenza_sego2022_large.yaml"
STUDY = ROOT / "workspace" / "studies" / "large-crossparadigm-comparison" / "study.yaml"


def _ids():
    return [str(c["id"]) for c in resolve_initial_state(Blueprint.load(LARGE))]


def _run(backend, n=4, seed=0):
    d = copy.deepcopy(Blueprint.load(LARGE).to_dict())
    d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
    return run_backend(Blueprint.from_dict(d), backend, n)


def test_large_blueprint_is_actually_large():
    assert len(_ids()) > 200  # ~225-cell epithelial block + immune pools + infected seed


def test_behavior_metrics_integrate_and_munk_moves_at_scale():
    cpm, munk = _run("cpm"), _run("munk")
    ids = _ids()
    reg = IdentityRegistry.from_backends(ids, munk.identity, cpm.identity,
                                         getattr(cpm, "residual", {}) or {})
    disp = metrics.per_cell_displacement_distance(munk.frames, cpm.frames, reg)
    msd = metrics.msd_distance(munk.frames, cpm.frames, reg)
    ng = metrics.neighbour_graph_distance(munk.frames, cpm.frames, reg)
    assert np.isfinite(disp["mean"]) and np.isfinite(msd) and np.isfinite(ng["mean"])
    # center-based now MOVES on the big tissue: its own MSD curve rises above ~0.
    assert np.nanmax(metrics.msd_curve(munk.frames)) > 0.1


def test_study_yaml_wellformed_and_uses_large_preset():
    s = yaml.safe_load(STUDY.read_text())
    assert s["baseline"][0]["composite"].endswith("sego_crossparadigm_large")
    assert {r["name"] for r in s["readouts"]} >= {
        "msd_distance", "neighbour_graph_distance", "front_speed_distance"}
    assert "PENDING_FILL" not in STUDY.read_text()
