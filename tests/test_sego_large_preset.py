"""The scaled-up Sego 2022 preset: a biology-preserving ~2.4x linear scale-up of
the canonical ``influenza_sego2022.yaml`` -- a bigger tissue, the SAME mechanism.

Guards three things:
  1. the LARGE blueprint loads, compiles, and runs a few coupled steps on all
     three backends, producing the expected cell types and ~225 epithelial cells;
  2. the thin preset composite ``sego_crossparadigm_large`` is registered and
     discoverable in the workspace catalog (the way the Studio surfaces it);
  3. the only differences from the canonical blueprint are geometry/counts --
     every other section (cell_types, adhesion, fields, rules, immune, ODE) is
     copied verbatim, so the biology is unchanged.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compare.runner import run_backend
from viva_multicell_blueprint.core import build_core

ROOT = Path(__file__).resolve().parents[1]
SEGO = ROOT / "workspace" / "blueprints" / "influenza_sego2022.yaml"
SEGO_LARGE = ROOT / "workspace" / "blueprints" / "influenza_sego2022_large.yaml"


@pytest.fixture(scope="module")
def bp():
    return Blueprint.load(SEGO_LARGE)


def test_large_blueprint_geometry(bp):
    assert bp.domain["extent_um"] == [120, 120]
    assert bp.domain["grid_resolution_um"] == 1 and bp.domain["boundary"] == "noflux"
    block = bp.initial_state["block"]
    assert block["shape"] == [15, 15] and block["spacing"] == 5 and block["size"] == 25
    assert block["origin"] == [25, 25]                       # re-centered in the 120x120 domain
    # ODE coupling count tracks the enlarged block
    assert bp.systemic_ode["coupling"]["num_epithelial"] == 225


@pytest.mark.parametrize("backend", ["cpm", "vertex", "munk"])
def test_large_blueprint_runs_on_all_backends(bp, backend):
    # Small step count: the larger domain means more cells + slower steps, so keep
    # the suite fast -- a few coupled steps is enough to prove it compiles + runs.
    run = run_backend(bp, backend, n_steps=3)
    assert len(run.frames) == 4                              # t0 + 3 coupled steps
    counts = Counter(run.frames[0]["types"].values())
    # 15x15 = 225 epithelial cells at t0, with a 5% infected seed (-> ~11 I)
    assert counts["H"] + counts["I"] == 225
    assert counts["I"] == 11
    # the three latent immune margin pools are pre-placed
    assert counts["M_latent"] == 12 and counts["NK_latent"] == 10 and counts["CD8_latent"] == 7


def test_large_preset_composite_is_registered():
    # The Studio catalog surfaces startable composites from the @composite_generator
    # registry; the thin preset must resolve under its canonical id and build.
    import viva_multicell_blueprint.composites  # noqa: F401 — fire the decorators
    from process_bigraph.composite_generator import _cs, build_generator

    sid = "viva_multicell_blueprint.composites.sego.sego_crossparadigm_large"
    spec = _cs.get(sid)
    assert spec is not None, "sego_crossparadigm_large not registered under its id"
    doc = build_generator(spec, overrides={"backend": "cpm"})
    assert isinstance(doc, dict) and "state" in doc
    # the workspace's own processes it addresses must resolve in build_core()
    assert "SpatialVideoStep" in build_core().link_registry


def test_large_preset_defaults_to_large_blueprint():
    # Calling the generator with no blueprint must start from the LARGE tissue.
    from viva_multicell_blueprint.composites.sego import sego_crossparadigm_large
    from process_bigraph import Composite

    core = build_core()
    doc = sego_crossparadigm_large(core=core, backend="cpm")
    comp = Composite({"state": doc["state"]}, core=core)
    comp.run(2.0)                                           # types store populates at run time
    counts = Counter(comp.state["types"].values())
    # epithelial lineage {H,I,D} is conserved (transitions stay within it) -> 225 cells,
    # proving the no-blueprint default started from the LARGE tissue, not the 36-cell one.
    assert counts["H"] + counts["I"] + counts["D"] == 225


def test_only_geometry_and_counts_differ_from_canonical():
    """Biology-preservation guard: every section except domain/initial_state and the
    ODE coupling COUNT is byte-for-byte identical to the canonical blueprint."""
    base = yaml.safe_load(SEGO.read_text())
    large = yaml.safe_load(SEGO_LARGE.read_text())
    for section in ("cell_types", "adhesion", "fields", "rules", "mechanics_mapping", "immune"):
        assert large[section] == base[section], f"{section} drifted -- biology must be identical"
    # the ODE block differs ONLY in the coupling count (eta numerator)
    assert large["systemic_ode"]["params"] == base["systemic_ode"]["params"]
    assert base["systemic_ode"]["coupling"]["num_epithelial"] == 36
    assert large["systemic_ode"]["coupling"]["num_epithelial"] == 225
    base_coupling = dict(base["systemic_ode"]["coupling"]); base_coupling.pop("num_epithelial")
    large_coupling = dict(large["systemic_ode"]["coupling"]); large_coupling.pop("num_epithelial")
    assert base_coupling == large_coupling


def test_served_large_copy_matches_source():
    """The Studio serves a copy of the large blueprint at reports/assets/ (the
    blueprints dir isn't web-served). Guard against it drifting from the source."""
    src = yaml.safe_load(SEGO_LARGE.read_text())
    served = yaml.safe_load((ROOT / "workspace" / "reports" / "assets" / "sego_blueprint_large.yaml").read_text())
    assert served == src, "reports/assets/sego_blueprint_large.yaml drifted; re-copy it"
