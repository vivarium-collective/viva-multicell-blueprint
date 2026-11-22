# A tour of multicell-blueprint

Setup is in [`docs/first-run-agent-guide.md`](docs/first-run-agent-guide.md); this page
walks the workspace from inputs to reports. Run `bash scripts/serve.sh` and follow along
in the workbench.

## 1. Inputs

- **Blueprints** — `workspace/blueprints/`. `influenza_sego2022.yaml` is the canonical
  Sego 2022 influenza model (50x50 patch, 36-cell epithelium); `influenza_sego2022_large.yaml`
  is the ~2.4x scale-up (120x120, ~225 cells), same mechanism. Blueprints use the v2
  sectioned format (schema and loader in `viva_multicell_blueprint/blueprint/`);
  `subcellular_ode_demo.yaml` is the growth/division example for the subcellular layer.
- **References and datasets** — `workspace/references/` (papers, bibliography, claims) and
  `workspace/datasets/`. Add with `scripts/add-reference.sh` / `scripts/add-dataset.sh`
  or the workbench's Workspace inputs tab.
- **Modules** — the mechanics backends and the subcellular solver come from imported
  repos declared in `workspace.yaml` (`viva-cpm`, `vivarium-tyssue`, `viva-munk`,
  `viva-grammar`, `viva-tellurium`). Manage them from the Registry tab.

## 2. The composite: `blueprint_pipeline`

`blueprint_pipeline` (`viva_multicell_blueprint/composites/pipeline.py`) is the one
canonical composite and the Studio entrypoint. Given a blueprint and a `backend`
(`cpm`, `vertex` or `munk`) it compiles the blueprint (`compile/`) into a process-bigraph
document: shared field, rule, systemic-ODE and immune-recruitment processes
(`mechanism/`), an optional per-cell subcellular layer, and exactly one mechanics node
(`backends/`). `sego_crossparadigm` and `sego_crossparadigm_large` are thin aliases that
default to the two Sego blueprints. Open any of them in the Composites tab to see the
wiring in the loom viewer, or in **Blueprint Studio** to edit and run a blueprint.

`docs/mechanics.md` explains each backend and the vertex calibration.

## 3. The investigation and its studies

One investigation, `influenza-crossparadigm` (`workspace/investigations/`), asks whether
one backend-neutral blueprint runs faithfully on three substrates. Its five studies live in
`workspace/studies/<slug>/study.yaml`; all use `blueprint_pipeline` as the baseline:

| Study | Question |
|---|---|
| `blueprint-pipeline-fidelity` | Does the pipeline reproduce the blueprint contract and t=0 state on every backend? |
| `cross-paradigm-equivalence` | Are the substrates equivalent on cell fate and on exact position, and does the test have power? |
| `substrate-behavior-at-scale` | On the large tissue, how far does each substrate sit from the CPM reference? |
| `dose-response-crossparadigm` | Do the substrates agree on the sign of the dose response to viral load? |
| `subcellular-layer-grow-divide` | Does the tellurium ODE layer grow and divide cells on all three backends? |

Each study has a `phase:` (Design, Build, Simulate, Evaluate, Decide) and follows the
v4 narrative spine: 14 sections in three layers, from the executive `report:` and
`study_card:` down to `tests:`, `readouts:` and `conclusion_verdicts:`. The schemas in
`.pbg/schemas/` define them and `scripts/lint-workspace.py` enforces them. Scaffold or
extend with `/viva:study <slug>` and `/viva:investigate <slug>`; seed a child study from a
Decide-phase proposal with `/viva:study seed-from-followup <parent>/<proposal_id>`.

## 4. Reports

- The published read-only workbench (badge in the README) shows every composite, study
  and result.
- `/viva:report` rebuilds the dashboard and per-investigation reports into
  `workspace/reports/`; the hosted copies are published on merge to `main`.
- Visualizations live in `viva_multicell_blueprint/viz/`; generate new ones with `/viva:viz`.

## When stuck

1. `python3 scripts/lint-workspace.py` — most workspace-shape issues surface here.
2. See Troubleshooting in `docs/first-run-agent-guide.md`.
3. File an issue at <https://github.com/vivarium-collective/viva-superpowers/issues>.
