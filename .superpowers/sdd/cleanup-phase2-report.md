# Cleanup phase 2: studies consolidated 9 -> 5

Branch `chore/streamline-consolidation`. All study.yaml files validate against `.pbg/schemas/study.schema.json`; investigation validates against `investigation.schema.json`.

## 9 -> 5 mapping as executed

| New study | Folded from | Notes |
|---|---|---|
| `blueprint-pipeline-fidelity` (S1) | `initial-state-fidelity` (dir renamed, spine), `bp-01-blueprint-schema` (yaml merged in, dir removed) | Design/Build contract (versioned schema, canonical/backend split, validator, 3 worked examples) + Simulate/Evaluate t0 numbers. Behavior tests: 3 contract + 4 t0. Vertex t0 numbers use the current 2026-11-05 reading (n_sheet 48, n_overlay 0, 12 immune attached, sheet RMS 1.60 um, immune attach RMS 4.11 um), matching tests/test_study_initial_state_fidelity.py; the superseded 36+12-overlay reading is kept as a metrics note. |
| `cross-paradigm-equivalence` (S2) | `crossparadigm-equivalence` (dir renamed, spine), `flu-lung-crossparadigm`, `substrate-divergence-counterfactual` | Three arms in one study: equivalence grid (TOST + run-cluster CI + cell-class split), N=400/N=150 same-backend null table (kept in full), mechanics-dominated positive control (explicit `controls:` entry kind=positive, own simulation_set `counterfactual-divergence`, own finding F3, own viz and results file). |
| `substrate-behavior-at-scale` (S3) | `large-crossparadigm-comparison` (dir renamed, spine), `adhesion-calibration`, `vertex-cpm-fluctuation-match` | Calibration folded in as a methods section (`description:` holds the term-by-term definition and fitting procedure, with docs/mechanics.md pointer) + simulation_set + findings. The unrun fluctuation-match scaffold became `design_pivot_required` (VCF-01, status open), a `followup_proposals` entry and 3 `implementation_requirements` (all planned). Not a standalone study. |
| `dose-response-crossparadigm` (S4) | kept | Composite re-pointed only (plus nothing else). |
| `subcellular-layer-grow-divide` (S5) | new | See below. |

Every kept/merged study baseline and simulation_set/variant composite is now `viva_multicell_blueprint.composites.pipeline.blueprint_pipeline`. S3 gets the large blueprint via `params.blueprint: workspace/blueprints/influenza_sego2022_large.yaml` (and the neutral sheet via a second baseline entry). No `sego_crossparadigm` string remains in any study.

## Where evidence ended up (all moved with `git mv`, none deleted or edited)

- `cross-paradigm-equivalence/results/equivalence.json` (from crossparadigm-equivalence)
- `cross-paradigm-equivalence/results/counterfactual-equivalence.json` (from substrate-divergence-counterfactual/results/equivalence.json; contents byte-identical, so its internal `"study"` field still holds the old slug)
- `cross-paradigm-equivalence/viz/`: equivalence-fate*, equivalence-spatial* (own); divergence*, dynamics-overlay, spatial-playback (from flu-lung); counterfactual-equivalence-spatial*, counterfactual-spatial-playback (from the counterfactual, prefixed `counterfactual-` to avoid name collision)
- `substrate-behavior-at-scale/results.json` (unchanged location), `results/vertex_mechanics_calibration.json` (from adhesion-calibration), viz: spatial-playback-cpm-large (own), adhesion-sweep, mechanics-calibration, neutral-sheet-playback (from adhesion-calibration)
- `blueprint-pipeline-fidelity/viz/`: t0-residuals, t0-side-by-side
- `dose-response-crossparadigm/viz/dose-response.html` unchanged
- Numbers from the removed study.yaml files (superseded readings included) were carried into the merged files' result/simulation_set/findings; the removed originals remain in git history.

## S5 subcellular-layer-grow-divide

Baseline `blueprint_pipeline` with `blueprint: workspace/blueprints/subcellular_ode_demo.yaml`, variants vertex and munk. Claims: per-cell ODE species evolve, target_size actuation grows cells, division with lineage on all three backends; cell counts cpm 4->8->16, vertex 4->8->16, munk 4->6->8->16; one S>40 crossing -> one division per starting cell (4->8) in the first 7 ticks. Three behavior tests mapped to tests/test_subcellular_demo_e2e.py. Honest evidence status stated in `limitations`, the simulation_set metrics (`evidence_source`) and the report: the evidence is the pytest e2e run, NOT a committed runs.db entry, and no visualization is attached. Also flags the CPM divide-all-over-threshold caveat and that cross-backend equivalence of growth/division statistics is untested. gate_status `partial`, confidence Investigating. The counts 4->8->16 etc. are taken from the task brief; the e2e test itself asserts growth, division, lineage and the 4->8 first-wave count (it passes), but only prints the full progression.

## Investigation rewrite (`influenza-crossparadigm/investigation.yaml`)

- Slug/dir unchanged (renaming would orphan the feedback files and review references); title/question/hypothesis broadened to blueprint -> `blueprint_pipeline` -> three substrates + pluggable subcellular layer, Sego influenza as the worked example.
- `description`: all `Task NN` refs and the "REVISED 2026-10-28 / (was 50)" and "--- 2026-10-27 framing / Prior reduced-scale reading" changelog-style blocks removed; rewritten around the five studies, three substrates (center-based no longer an appendix; added as a competing framework), single pipeline composite.
- `at_a_glance`, `acceptance_criteria`, `roadmap`, `claims` evidence, `decisions_needed` rewritten for the 5 slugs; added a decision for the subcellular layer; `members:` updated and a `studies:` list added (the old file had only `members:`).
- Kept: claims/stages/guardrails/proposed_studies/figure_plan (F1 retitled "three substrates"), glossary (added blueprint_pipeline, subcellular layer).
- Deliberately NOT edited: `feedback/`, `feedback-response.md`, `workspace/references/reviews/*`, `docs/superpowers/*` (historical documents that name the old slugs).

## Code / tests touched

- `tests/test_large_crossparadigm_study.py`: STUDY path -> `substrate-behavior-at-scale`; assertion now checks name, `blueprint_pipeline` composite and `influenza_sego2022_large.yaml` blueprint param (was `endswith("sego_crossparadigm_large")`).
- Other study tests do not read study.yaml; only docstrings updated for the new slugs (test file names unchanged).
- `scripts/regen_equivalence.py`: both arms write into `cross-paradigm-equivalence` (counterfactual files prefixed `counterfactual-`, results `counterfactual-equivalence.json`, JSON gains `"arm"`). `scripts/render_study_figures.py`: renderer keys/paths remapped (new `substrate-behavior-at-scale` renderer = adhesion + large). `scripts/calibrate_vertex_mechanics.py` OUT path, `compile/to_vertex.py` + `docs/mechanics.md` path refs, `workbench_viewers.py` (Studio anchored to `cross-paradigm-equivalence`), `workspace/blueprints/mechanics_sheet.yaml` comment, assorted docstrings/comments.
- Schema fix: S3's legacy `relation: leads-to` migrated to canonical `builds-on` (it was schema-invalid before).

## Test results

Run with the main checkout's venv and `PYTHONPATH` = this worktree: `tests/test_sego_blueprint.py tests/test_large_crossparadigm_study.py tests/test_subcellular_demo_e2e.py tests/test_blueprint_canonical.py tests/test_study_figures.py` -> 32 passed. Full suite: 355 passed, 2 skipped.

## Flags

- Figures were NOT re-rendered (no regen scripts run); existing HTML is moved, not regenerated, so the next `render_study_figures.py`/`regen_equivalence.py` run will write to the new locations.
- The `viz` entries in S2 point at moved files whose embedded titles still say "CPM | vertex" in places; not edited.
- `.pbg` investigation schema may or may not recognize `studies:`; it validated with 0 errors.
