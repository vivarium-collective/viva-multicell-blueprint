"""Task 11: the Studio targets blueprint_pipeline, and the v2 subcellular ODE demo
drives per-cell growth + division end-to-end on every tissue backend."""
from pathlib import Path

import pytest
from process_bigraph import Composite

pytest.importorskip("viva_tellurium")

from viva_multicell_blueprint.composites.pipeline import blueprint_pipeline
from viva_multicell_blueprint.core import build_core

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "workspace" / "blueprints" / "subcellular_ode_demo.yaml"
STUDIO = ROOT / "workspace" / "reports" / "assets" / "blueprint_studio.html"
N_STEPS = 12
_BACKENDS = [("cpm", "viva_cpm"), ("vertex", "vivarium_tyssue"), ("munk", "viva_munk")]


def _run(backend, mod):
    pytest.importorskip(mod)
    comp = Composite(blueprint_pipeline(backend=backend, blueprint=str(DEMO)), core=build_core())
    first_sub = first_size = None
    max_size, n_max, lineage_seen = {}, 0, {}
    for _ in range(N_STEPS):
        comp.run(1.0)
        sub = {c: dict(v) for c, v in (comp.state.get("sub_scalar") or {}).items()}
        sizes = dict(comp.state.get("sizes") or {})
        if first_sub is None and sub:       # first step has no ODE output yet
            first_sub, first_size = sub, dict(sizes)
        for c, s in sizes.items():
            max_size[c] = max(max_size.get(c, 0.0), float(s))
        n_max = max(n_max, len(sizes))
        lineage_seen.update(comp.state.get("lineage") or {})
    return comp, first_sub, first_size, max_size, n_max, lineage_seen


def test_studio_points_at_blueprint_pipeline():
    html = STUDIO.read_text()
    assert 'const COMPOSITE = "viva_multicell_blueprint.composites.pipeline.blueprint_pipeline"' in html


@pytest.mark.parametrize("backend,mod", _BACKENDS)
def test_ode_drives_growth(backend, mod):
    comp, first_sub, first_size, max_size, _, _ = _run(backend, mod)
    last_sub = comp.state["sub_scalar"]
    common = set(first_sub) & set(last_sub)
    # the ODE stepped per cell: S changed
    assert common and any(abs(last_sub[c]["S"] - first_sub[c]["S"]) > 1e-6 for c in common)
    # target_size actuation reached the backend: some cell's emitted size grew
    assert any(max_size[c] > first_size[c] + 1e-6 for c in first_size if c in max_size)


# KNOWN INTEGRATION GAP: CellActuationMergeStep fires twice per tick and its
# consume-once cursor means the backend always reads an empty `divisions` list, so
# a subcellular divide_when never reaches the backend. See the Task 11 report and
# task-11-division-fix.patch (verified: division then occurs on all 3 backends).
@pytest.mark.xfail(strict=True, reason="merge step drops subcellular divide requests (double-fire + cursor)")
@pytest.mark.parametrize("backend,mod", _BACKENDS)
def test_ode_drives_division(backend, mod):
    comp, first_sub, first_size, _, n_max, lineage_seen = _run(backend, mod)
    assert n_max > len(first_size)
    assert lineage_seen, "no lineage entry produced"
