"""Workspace derived-scalar computers for behavior-test grading.

The discovery-engine evaluator (``viva-compute-outcomes``) resolves a behavior test's
``measure: {kind: derived_scalar, path: <name>}`` to a function registered here, calls it
with ``(reader, test, ws_root)``, and grades the returned scalar against the test's
``pass_if``. These reduce a run's emitted observables (persisted by the SQLiteEmitter) to
the single number each composition-atlas claim is pinned on.
"""
from __future__ import annotations

from typing import Any

try:  # the decorator is optional (declares which observables a scalar reads)
    from discovery_engine.study.evaluator.workspace_plugins import declares_sources
except Exception:  # pragma: no cover
    def declares_sources(*_a):  # type: ignore
        def _d(fn):
            return fn
        return _d


def _values(reader: Any, observable: str) -> list[float]:
    """The time-ordered values of one emitted observable, as plain floats."""
    try:
        df = reader.series(observable)
        cols = list(getattr(df, "columns", []))
        for c in (observable, "value", "element", cols[-1] if cols else None):
            if c and c in cols:
                return [float(v) for v in df[c].to_list() if v is not None]
    except Exception:
        pass
    try:
        return [float(r[observable]) for r in reader.rows([observable])
                if r.get(observable) is not None]
    except Exception:
        return []


@declares_sources("probe_a")
def atlas_fate_transitions(reader, test, ws_root) -> float:
    """Number of times the probed Boolean signal (probe_a) crosses the 0.5 cutoff.

    For the oscillator / triple cell, probe_a is the fate 'divide' activity — crossings
    count the metabolism↔fate oscillation (a latched signal gives 0)."""
    vals = _values(reader, "probe_a")
    hi = [v > 0.5 for v in vals]
    return float(sum(1 for a, b in zip(hi, hi[1:]) if a != b))


@declares_sources("probe_a")
def atlas_product_final(reader, test, ws_root) -> float:
    """Final value of the probed ODE product (probe_a = metab/P) — did the catalysed
    product accumulate?"""
    vals = _values(reader, "probe_a")
    return float(vals[-1]) if vals else 0.0


@declares_sources("probe_b")
def atlas_signal_growth(reader, test, ws_root) -> float:
    """Growth of the probed driver signal over the run (probe_b final − initial). For the
    catalysis study, probe_b is the particle C count — it must build for the coupling to
    show."""
    vals = _values(reader, "probe_b")
    return float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0


@declares_sources("n_divisions")
def atlas_divisions_final(reader, test, ws_root) -> float:
    """Final cumulative fate-driven division count (fateful tissue)."""
    vals = _values(reader, "n_divisions")
    return float(vals[-1]) if vals else 0.0


@declares_sources("n_cells")
def atlas_cells_min(reader, test, ws_root) -> float:
    """Minimum cell count over the simulated run — a non-collapsing tissue stays at or above
    its initial size (fateful tissue). The leading emit (index 0) is the pre-run document
    state (n_cells=0 before the first update) and is skipped."""
    vals = _values(reader, "n_cells")
    sim = vals[1:] if len(vals) > 1 else vals
    return float(min(sim)) if sim else 0.0


@declares_sources("confined_fraction")
def atlas_confined_min(reader, test, ws_root) -> float:
    """Minimum confined fraction over the simulated run — the fraction of particles inside
    their own cell's boundary (bounded-particles confinement). The leading pre-run emit
    (index 0, confined_fraction=0.0 before any particles) is skipped."""
    vals = _values(reader, "confined_fraction")
    sim = vals[1:] if len(vals) > 1 else vals
    return float(min(sim)) if sim else 0.0


@declares_sources("total_molecules")
def atlas_total_molecules_final(reader, test, ws_root) -> float:
    """Final total molecule count across the scene's cells — the mixed-formalism composite
    produced live molecular state (scene-model)."""
    vals = _values(reader, "total_molecules")
    return float(vals[-1]) if vals else 0.0


@declares_sources("consensus_spread")
def atlas_spread_max(reader, test, ws_root) -> float:
    """Maximum consensus spread over the run — the largest per-cell distance between the three
    backends' centroids. A small bound means CPM, vertex and munk stay coupled on one cell set."""
    vals = _values(reader, "consensus_spread")
    sim = vals[1:] if len(vals) > 1 else vals
    return float(max(sim)) if sim else 0.0


@declares_sources("committed_fraction")
def atlas_committed_final(reader, test, ws_root) -> float:
    """Final fraction of cells whose fate committed (divide > 0.5) — the end state of the
    area → metabolism → fate cross-scale coupling in the consensus tissue."""
    vals = _values(reader, "committed_fraction")
    return float(vals[-1]) if vals else 0.0


@declares_sources("interface_breaches")
def atlas_breaches_max(reader, test, ws_root) -> float:
    """Max per-step interface breaches (mesenchyme cells reflected off the epithelial floor). A
    small bound means the interface holds — the two tissues couple without the mesenchyme
    flooding through."""
    vals = _values(reader, "interface_breaches")
    sim = vals[1:] if len(vals) > 1 else vals
    return float(max(sim)) if sim else 0.0


def register_derived_scalars(reg: dict) -> None:
    reg["atlas_fate_transitions"] = atlas_fate_transitions
    reg["atlas_product_final"] = atlas_product_final
    reg["atlas_signal_growth"] = atlas_signal_growth
    reg["atlas_divisions_final"] = atlas_divisions_final
    reg["atlas_cells_min"] = atlas_cells_min
    reg["atlas_confined_min"] = atlas_confined_min
    reg["atlas_total_molecules_final"] = atlas_total_molecules_final
    reg["atlas_spread_max"] = atlas_spread_max
    reg["atlas_committed_final"] = atlas_committed_final
    reg["atlas_breaches_max"] = atlas_breaches_max
