"""Run each composition-atlas study's composite, persist a run store, and grade it.

For each study: build the composite, attach a SQLiteEmitter capturing the observables the
study's behavior tests read, run it, register the run in the study's runs.db, then
``sync`` (record_runs + compute_outcomes) so the committed behavior tests are graded against
the real run and the study earns a computed verdict.

Usage:
  PYTHONPATH=$(pwd) <venv>/bin/python scripts/run_atlas_studies.py [study-slug ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from process_bigraph import Composite
from process_bigraph.emitter import emitter_from_wires

from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.composites.coupled_subcellular import (
    coupled_fate_oscillator, coupled_particle_catalysis, coupled_triple_cell)
from viva_multicell_blueprint.composites.fateful_tissue import fateful_tissue

from discovery_engine.study.run_registry import register_run
from viva_superpowers.study_outcomes import record_runs
from viva_superpowers.study_evaluator import compute_outcomes

STUDIES = Path("workspace/studies")
WS_ROOT = str(Path(".").resolve())  # compute_outcomes needs ws_root to load derived scalars
RUN_ID = "baseline"


def _sync_with_ws(study_dir: Path) -> dict:
    """sync() but passing ws_root so workspace derived-scalar graders are loaded
    (viva_superpowers.study_outcomes.sync hardcodes compute_outcomes(study_dir) with no
    ws_root, which silently loads no workspace evaluators)."""
    summary = record_runs(study_dir)
    summary["computed"] = compute_outcomes(study_dir, ws_root=WS_ROOT)
    _mirror_outcomes(study_dir)  # keep the legacy gate reader in sync with the verdict
    try:
        from viva_superpowers.simulation_set import populate_simulation_set
        summary["simulation_set"] = populate_simulation_set(study_dir)
    except Exception as exc:  # noqa: BLE001
        summary["simulation_set"] = {"error": str(exc)}
    try:
        from viva_superpowers.study_verdict import write_gate_evaluator
        summary["gate"] = {"changed": write_gate_evaluator(study_dir)}
    except Exception as exc:  # noqa: BLE001
        summary["gate"] = {"error": str(exc)}
    return summary


def _mirror_outcomes(study_dir: Path) -> None:
    """Mirror each run's code-graded ``computed_outcomes`` into ``outcomes`` so the gate
    evaluator (which reads ``runs[].outcomes`` via canonical_outcomes) agrees with the verdict
    (which reads ``computed_outcomes``) — two parallel schemas in the current tooling."""
    from ruamel.yaml import YAML
    sy = study_dir / "study.yaml"
    yaml = YAML(); yaml.preserve_quotes = True; yaml.width = 4096
    doc = yaml.load(sy.read_text(encoding="utf-8"))
    touched = False
    for run in doc.get("runs") or []:
        co = run.get("computed_outcomes") or {}
        mirror = {k: {"result": v.get("result")} for k, v in co.items()
                  if isinstance(v, dict) and v.get("result")}
        if mirror:
            run["outcomes"] = mirror
            touched = True
    if touched:
        with sy.open("w", encoding="utf-8") as fh:
            yaml.dump(doc, fh)

_C = "viva_multicell_blueprint.composites"
SPECS = {
    "coupled-fate-oscillator": dict(
        gen=coupled_fate_oscillator, params=dict(n_cells=1), steps=40,
        obs=["probe_a", "probe_b", "n_cells"],
        spec_id=f"{_C}.coupled_subcellular.coupled_fate_oscillator"),
    "coupled-particle-catalysis": dict(
        gen=coupled_particle_catalysis, params=dict(n_cells=1), steps=30,
        obs=["probe_a", "probe_b", "n_cells"],
        spec_id=f"{_C}.coupled_subcellular.coupled_particle_catalysis"),
    "coupled-triple-cell": dict(
        gen=coupled_triple_cell, params=dict(n_cells=1), steps=36,
        obs=["probe_a", "probe_b", "n_cells"],
        spec_id=f"{_C}.coupled_subcellular.coupled_triple_cell"),
    "fateful-tissue": dict(
        gen=fateful_tissue, params=dict(seed_per_cell=40), steps=22,
        obs=["n_divisions", "n_cells", "n_particles", "mean_metabolite"],
        spec_id=f"{_C}.fateful_tissue.fateful_tissue"),
}


def run_one(slug: str) -> None:
    spec = SPECS[slug]
    study_dir = STUDIES / slug
    store_dir = (study_dir / "runs" / RUN_ID).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)
    # fresh store each run (the SQLiteEmitter appends; a stale db collides on sim_id+step)
    db = store_dir / "history.db"
    if db.exists():
        db.unlink()

    core = build_core()
    doc = spec["gen"](core=core, **spec["params"])
    wires = {k: [k] for k in spec["obs"]}
    wires["global_time"] = ["global_time"]
    em = emitter_from_wires(wires, address="local:SQLiteEmitter")
    em["config"] = {**em.get("config", {}), "file_path": str(store_dir),
                    "db_file": "history.db", "simulation_id": f"{slug}-{RUN_ID}", "name": RUN_ID}
    doc["state"]["emitter"] = em

    comp = Composite({"state": doc["state"]}, core=core)
    t0 = time.time()
    comp.run(spec["steps"])
    t1 = time.time()

    # portable store path (relative to the worktree root, where we run from) so both
    # record_runs' RunReader.open and the evaluator's resolve_run_store find it
    rel_store = f"workspace/studies/{slug}/runs/{RUN_ID}/history.db"
    register_run(study_dir / "runs.db", RUN_ID, spec_id=spec["spec_id"], status="completed",
                 n_steps=spec["steps"], emitter_path=rel_store,
                 started_at=t0, completed_at=t1)
    result = _sync_with_ws(study_dir)
    print(f"  {slug}: ran {spec['steps']} steps in {t1 - t0:.1f}s  sync={result}")


def main(argv):
    slugs = argv or list(SPECS)
    for slug in slugs:
        if slug not in SPECS:
            print(f"unknown study {slug!r}; have {list(SPECS)}")
            continue
        print(f"== {slug} ==")
        run_one(slug)


if __name__ == "__main__":
    main(sys.argv[1:])
