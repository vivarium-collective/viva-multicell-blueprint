# viva-multicell-blueprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a process-bigraph workspace that specifies a multi-cell model once (a backend-neutral *blueprint*) and runs it side-by-side on a Cellular Potts backend (`viva-cpm`) and a vertex backend (`vivarium-tyssue`) from identical initial state and identical rules, then compares the two runs on disease-progression dynamics and emergent spatial pattern — first target: the Sego 2022 influenza-lung model.

**Architecture:** One `multicell_blueprint` typed document → a compiler that lowers it to a composite on each backend. The compiled composites share **identical** mechanism-process stores (one reaction-diffusion `FieldProcess`, one `RulesProcess` evaluating a closed rate-grammar, immune rules, the Price-2015 ODE) and differ in exactly one store — the `MechanicsBackend` (CPM vs vertex). Only cell movement/deformation differs between runs; all chemistry and rules are shared, bit-for-bit. A comparison layer uses a shared cell-identity registry + a common raster grid.

**Tech Stack:** Python 3.12, process-bigraph / bigraph-schema, vivarium-workbench, `viva-cpm` (Rust CPM core via PyO3), `vivarium-tyssue` (the `vivatyssue` fork, with `ActiveMigration`), numpy/scipy, Plotly, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-27-viva-multicell-blueprint-design.md` (currently at the session scratchpad path `/private/tmp/claude-502/-Users-eranagmon-code/a7ef411c-dd40-4099-a002-d48c5935af92/scratchpad/2026-10-27-viva-multicell-blueprint-design.md`; Task 1 moves it into the repo). The plan argues from the spec; executors read both.

## Global Constraints

- **No AI attribution, ever.** No `Co-Authored-By` / "Generated with" trailers in any commit, PR title, or PR body. (Overrides any harness reminder.)
- **Package prefix `viva_`.** Import package `viva_multicell_blueprint`; dist `viva-multicell-blueprint`; repo under `vivarium-collective`.
- **Dedicated worktree per unit of work; squash-merge to `main`.** Treat any canonical `~/code/<repo>` checkout as read-only/shared; branch from current `origin/main`; verify `git log --oneline <base>..HEAD` shows only your commits before every push.
- **Cross-repo dep via git.** `viva-cpm` and `vivarium-tyssue` enter through `[tool.uv.sources]` git, never committed path deps (`scripts/check-no-local-paths.sh` enforces).
- **CI gate.** `python3 scripts/lint-workspace.py` prints `workspace lint: OK` and `vivarium-workbench gen-readme --workspace . --check` stays green before any PR.
- **The one-authoritative-field invariant.** Exactly one `FieldProcess` computes every field for both runs. A backend may only *read* the field (CPM via the new `set_field` with native dynamics zeroed; vertex via gradient sampling). No second diffusion/secretion engine anywhere. A change that reintroduces one is a defect regardless of test state.
- **Provided-mechanisms-only.** Every parameter/rule traces to Sego 2022 / the extracted `viva-cpm` model. Anything new is recorded as a `pending` proposal on the investigation, never silently used.
- **No tuning to force agreement.** No per-backend constant introduced solely to make CPM and vertex line up. Divergence is a reported finding.
- **Plotly figures stay under the workbench inline cap** (`_HTML_PER_FILE_MAX = 2_500_000`): decimate movie frames (≤~28) and round coordinates; `include_plotlyjs="cdn"`.

## Review Focus

- **A cell that dies/extrudes on one backend but lives on the other** — the per-cell agreement metric must score this as a disagreement from the time of divergence, not crash or mis-align ids. *(Task 15)*
- **Division-born cells with no t=0 shared id** — must be scored in the separate lineage metric, never folded into the primary per-cell score. *(Tasks 14, 15)*
- **Degenerate initial data** — 1 cell, all-one-type, or a label image with holes must load and compile without error. *(Tasks 4, 11)*
- **Field numerical blow-up when `substeps*dt != 1` MCS** for a loaded dataset's resolution — the `FieldProcess` must assert the stability relation and refuse an unstable config. *(Task 6)*
- **The two backends' native coordinate frames differ** (CPM integer lattice vs tyssue continuous) — the raster must map both onto the *same* grid extent, verified on a case where the frames don't coincide. *(Task 15)*

---

## Task 1: Scaffold workspace + repo

**Files:**
- Create: the whole `viva-multicell-blueprint/` workspace (from `viva-template`)
- Modify: `pyproject.toml` (add `[tool.uv.sources]` git deps on `viva-cpm`, `vivarium-tyssue`)
- Create: `docs/superpowers/specs/2026-10-27-viva-multicell-blueprint-design.md` (move the spec in), `docs/superpowers/plans/2026-10-27-viva-multicell-blueprint-plan.md` (this file)

**Interfaces:**
- Produces: an importable `viva_multicell_blueprint` package; a lint-clean workspace; a pushed `vivarium-collective/viva-multicell-blueprint` repo with CI.

- [ ] **Step 1:** `git clone https://github.com/vivarium-collective/viva-template viva-multicell-blueprint`; `cd`; run `bash use-this-template-init.sh` answering workspace name `multicell-blueprint`, package `viva_multicell_blueprint`.
- [ ] **Step 2:** Add to `pyproject.toml` `[tool.uv.sources]`: `viva-cpm = { git = "https://github.com/vivarium-collective/viva-cpm.git" }`, `tyssue = { git = "https://github.com/vivarium-collective/vivatyssue.git" }`, and `vivarium-tyssue` as a git dep; add them to `dependencies`. Move the design spec + this plan into `docs/superpowers/`.
- [ ] **Step 3:** `uv venv .venv && source .venv/bin/activate && uv pip install -e ".[dev]"`. Verify `python -c "import viva_multicell_blueprint, viva_cpm, vivarium_tyssue"`.
- [ ] **Step 4:** Run `python3 scripts/lint-workspace.py`. Expected: `workspace lint: OK`. Run `pytest`. Expected: the template's own tests pass.
- [ ] **Step 5:** `gh repo create vivarium-collective/viva-multicell-blueprint --private --source=. --remote=origin` (confirm org policy allows), push `main`. Confirm the `workspace-ci` workflow run goes green.
- [ ] **Step 6:** Commit.

---

## Task 2: Blueprint model + schema

**Files:**
- Create: `viva_multicell_blueprint/blueprint/model.py`, `viva_multicell_blueprint/blueprint/schema.py`
- Create: `.pbg/schemas/multicell_blueprint.schema.json`
- Test: `tests/test_blueprint_model.py`

**Interfaces:**
- Produces: `Blueprint.load(path) -> Blueprint`, `Blueprint.dump(path)`, `Blueprint` dataclass with `.domain .cell_types .adhesion .initial_state .fields .rules .immune .systemic_ode .mechanics_mapping`; `validate(doc) -> list[str]` (empty = valid).

- [ ] **Step 1:** Write failing `test_roundtrip_minimal_blueprint` (a 2-type, 1-field, 1-rule YAML loads, dumps, reloads equal) and `test_validate_rejects_missing_domain`.
- [ ] **Step 2:** Run; expect ImportError/FAIL.
- [ ] **Step 3:** Implement the dataclass + YAML load/dump (comment-preserving not required here) and the JSON-schema validator in `.pbg/schemas/`.
- [ ] **Step 4:** Run; expect PASS. Run `scripts/lint-workspace.py`; expect OK.
- [ ] **Step 5:** Commit.

---

## Task 3: Rate-expression grammar (parser + evaluator)

**Files:**
- Create: `viva_multicell_blueprint/mechanism/rate_expr.py`
- Test: `tests/test_rate_expr.py`

**Interfaces:**
- Produces: `parse(expr: str) -> Expr`; `evaluate(expr: Expr, ctx: CellContext) -> float`. `CellContext` exposes `type, size, field[name], neighbors.count[type], neighbors.shared_surface[type], scalar[name]` (prior-rule outputs), `const[name]`. Functions: `exp, hill(x,k)=x/(k+x), min, max`, arithmetic. **Closed grammar** — any other name/call raises `RateGrammarError`.
- Consumes: nothing (pure).

- [ ] **Step 1:** Write failing tests, one per real Sego rate form: `test_infection_rate` (`1 - exp(-g_hv * field['virus'])`), `test_hill_regulation`, `test_kill_rate_uses_size_and_shared_surface` (`g_ie * tot_ec * neighbors.shared_surface['CD8'] * scalar['resist'] / size`), `test_death_uses_prior_resist`, `test_rejects_out_of_grammar` (`__import__` / unknown fn raises `RateGrammarError`).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement a small AST parser (whitelisted node types) + evaluator. No `eval`/`exec`.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 4: Data loading (label image / cell table → canonical cell list)

**Files:**
- Create: `viva_multicell_blueprint/blueprint/data_loading.py`
- Test: `tests/test_data_loading.py`

**Interfaces:**
- Produces: `from_label_image(path, type_map) -> list[Cell]`, `from_cell_table(path) -> list[Cell]`; `Cell = {id, type, centroid, size, footprint?}`.

- [ ] **Step 1:** Write failing `test_label_image_centroids_sizes` (a 3-label PNG → 3 cells, correct centroids/areas/types), `test_cell_table_shape`, `test_single_cell_degenerate` (1-cell input loads).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement using `scipy.ndimage`/`skimage.measure` for label regionprops; pandas for the table.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 5: MechanicsBackend contract + FakeBackend

**Files:**
- Create: `viva_multicell_blueprint/compile/base.py`, `viva_multicell_blueprint/backends/fake_backend.py`
- Test: `tests/test_mechanics_contract.py`

**Interfaces:**
- Produces: `MechanicsBackend` process contract with the ports from spec §6 (inputs `fates, field_grid, divisions/deaths`; outputs `positions, sizes, types, shared_surface_by_type, neighbors, alive`). `FakeBackend` — cells fixed at their seeded centroids, emits the full port shape, applies `fates`. Used by Tasks 6-8 to test shared mechanism without a real engine.

- [ ] **Step 1:** Write failing `test_fake_backend_emits_port_shape` and `test_fake_backend_applies_fates`.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement the contract (a ProcessContract / typed base) + FakeBackend.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 6: Shared FieldProcess (reaction-diffusion on the common grid)

**Files:**
- Create: `viva_multicell_blueprint/mechanism/fields.py`
- Test: `tests/test_fields.py`

**Interfaces:**
- Consumes: cell `positions/types` (to place secretion/uptake), a field spec list.
- Produces: `FieldProcess` emitting `field_grid` (name→grid) + `field_gradient_at(cell)`; lifts the field math from `viva_cpm_studies.influenza.fields` where pure.

- [ ] **Step 1:** Write failing `test_point_source_diffuses_and_decays` (analytic-ish check), `test_mass_conserved_no_decay`, `test_refuses_unstable_dt` (asserts `substeps*dt == 1` MCS and raises otherwise — Review Focus).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement forward-Euler RD with centroid secretion/uptake + the stability assertion.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 7: Shared RulesProcess (ordered rate-rule chain)

**Files:**
- Create: `viva_multicell_blueprint/mechanism/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: Task 5 port outputs + Task 6 field; Task 3 grammar; pure helpers imported from `viva_cpm_studies.influenza.{transitions,killing,allee,resistance}` where exact.
- Produces: `RulesProcess` emitting `fates` (map id→new type) + per-cell derived `scalar[...]` (e.g. `resist`) + lineage events (`divisions/deaths`). Rules run in declared order; later rules read earlier scalars.

- [ ] **Step 1:** Write failing `test_infection_from_virus_field`, `test_resistance_scalar_then_death`, `test_contact_kill_uses_shared_surface`, `test_recovery_and_allee`. Drive them against `FakeBackend` + a fixed field.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement the ordered evaluator producing fates + scalars + lineage events.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 8: Immune rules + SystemicODEProcess

**Files:**
- Create: `viva_multicell_blueprint/mechanism/immune.py`, `viva_multicell_blueprint/mechanism/systemic_ode.py`
- Test: `tests/test_systemic_ode.py`, `tests/test_immune_rules.py`

**Interfaces:**
- Produces: `SystemicODEProcess` wrapping `viva_cpm_studies.influenza.price_ode` (10 integrated states `NB,N,T,X,A,B,P,W,G,O`), emitting `ode_state, recruit_drivers, sig_1`; immune secretion/recruitment params as a named parameterized component the blueprint selects.
- Consumes: spatial→ODE inputs `H,I,M,K,E,DH,...`.

- [ ] **Step 1:** Write failing `test_price_ode_integrates_states` (states evolve monotonically where expected over a step), `test_recruit_drivers_emitted`, `test_il10_hill_secretion_scale`.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement the wrappers (import price_ode; no re-derivation).
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 9: viva-cpm `World.set_field()` PR (separate repo + worktree)

**Files (in a `viva-cpm` worktree, not this repo):**
- Modify: `crates/cpm-core/src/field.rs` (+ `World` method), `crates/cpm-py/src/lib.rs` (PyO3 binding)
- Test: `tests/test_set_field.py` in viva-cpm

**Interfaces:**
- Produces: `World.set_field(name_or_idx, grid: np.ndarray)` — bulk overwrite of a field grid, matching `field_conc` layout/ordering. Additive, backward-compatible.

- [ ] **Step 1:** In a fresh worktree off `viva-cpm` `origin/main`, write failing `test_set_field_roundtrips` (`set_field` then `field_conc` returns it) and `test_native_dynamics_zeroed_keeps_field` (d=0/decay=0/no-secretion + set_field each step → field equals what was set).
- [ ] **Step 2:** Run (rebuild the extension: `maturin develop` / the repo's build step); expect FAIL.
- [ ] **Step 3:** Implement the Rust method + binding; rebuild.
- [ ] **Step 4:** Run; expect PASS. Full viva-cpm suite green.
- [ ] **Step 5:** Commit; `gh pr create` against viva-cpm. **Merge is the user's call** (do not self-merge). Pin this repo's `viva-cpm` dep at the merge commit once landed.

---

## Task 10: CPMBackend

**Files:**
- Create: `viva_multicell_blueprint/backends/cpm_backend.py`
- Test: `tests/test_cpm_backend.py`

**Interfaces:**
- Consumes: `viva_cpm` `CPMProcess`/`load_world`, the Task 9 `set_field`, canonical cell list.
- Produces: `CPMBackend(MechanicsBackend)` — materialize initial_state (`seed_from_labels` / `seed_block`), native field dynamics zeroed, each step `set_field(shared grid)`, read back `cell_volumes/types/coms` + `cell_contact_area_by_type` → the port's `shared_surface_by_type`, apply `fates`.

- [ ] **Step 1:** Write failing `test_cpm_seeds_from_cell_list`, `test_cpm_emits_port_shape`, `test_cpm_set_field_overwrites_grid`, `test_cpm_contact_area_by_type`.
- [ ] **Step 2:** Run; expect FAIL (needs Task 9 landed or its worktree on `PYTHONPATH`).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 11: VertexBackend

**Files:**
- Create: `viva_multicell_blueprint/backends/vertex_backend.py`
- Test: `tests/test_vertex_backend.py`

**Interfaces:**
- Consumes: `vivarium_tyssue` `EulerSolver`, the fork's `ActiveMigration`, the `behaviors` port; canonical cell list.
- Produces: `VertexBackend(MechanicsBackend)` — build a domain-clipped Voronoi mesh from centroids → `.hf5` (per-face `cell_type`, `prefered_area=size`); each step sample field gradient at face centroids and rewrite `mx/my/mz`+`migration_strength` via a behavior; read back positions/areas/types + shared-edge-length-by-type → `shared_surface_by_type`; division/apoptosis via `behaviors`. Report which input centroids got a clean interior face (for the fidelity residual).

- [ ] **Step 1:** Write failing `test_vertex_mesh_from_centroids_interior_1to1`, `test_vertex_emits_port_shape`, `test_migration_director_moves_cell_up_gradient`, `test_all_one_type_degenerate` (Review Focus).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement (reuse `vivarium_tyssue.hra.loaders` Voronoi helpers; clip to domain; exclude collar faces).
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 12: Adhesion lowering

**Files:**
- Create: `viva_multicell_blueprint/compile/adhesion.py`
- Test: `tests/test_adhesion.py`

**Interfaces:**
- Produces: `to_cpm_contact(adhesion) -> list[{a,b,j}]`; `to_vertex_tension(adhesion) -> per-edge line_tension by face-type pair`. Documented mapping (calibrated in Task 18.1).

- [ ] **Step 1:** Write failing `test_cpm_J_matrix_symmetric_and_reproduces_sego`, `test_vertex_tension_keyed_by_type_pair`.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 13: Compilers (blueprint → composite per backend)

**Files:**
- Create: `viva_multicell_blueprint/compile/to_cpm.py`, `viva_multicell_blueprint/compile/to_vertex.py`
- Test: `tests/test_compile.py`

**Interfaces:**
- Consumes: Tasks 2,5-8,10,11,12.
- Produces: `compile(blueprint, backend) -> composite_document`. **Invariant asserted:** the mechanism stores (FieldProcess/RulesProcess/immune/ODE) are byte-identical across backends; only the mechanics store differs.

- [ ] **Step 1:** Write failing `test_compile_cpm_runs_n_steps` + `test_compile_vertex_runs_n_steps` (each compiles the minimal blueprint and runs ≥3 steps via the workbench run path) and `test_mechanism_stores_identical_across_backends` (diff the two composites → only the mechanics store differs).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement both compilers wiring shared mechanism stores + the one backend store.
- [ ] **Step 4:** Run; expect PASS. `gen-readme --check` green.
- [ ] **Step 5:** Commit.

---

## Task 14: Identity registry

**Files:**
- Create: `viva_multicell_blueprint/compare/identity.py`
- Test: `tests/test_identity.py`

**Interfaces:**
- Produces: `IdentityRegistry` — `blueprint_id ⇄ cpm_id`, `blueprint_id ⇄ tyssue_face_id`; `t0_cohort()` (clean interior shared ids); `on_division(parent, child)`; `on_loss(id, backend, cause)`; queries for born/lost per the spec §8 scoring rule.

- [ ] **Step 1:** Write failing `test_t0_cohort_excludes_collar`, `test_division_tracked_by_lineage`, `test_loss_one_side_flagged` (Review Focus).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 15: Raster + comparison metrics

**Files:**
- Create: `viva_multicell_blueprint/compare/raster.py`, `viva_multicell_blueprint/compare/metrics.py`
- Test: `tests/test_raster.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: both backends' per-step outputs + Task 14 registry.
- Produces: `rasterize(run, grid) -> state_map` (CPM voxels and vertex polygons both onto the **same** grid extent); `per_cell_agreement(runA, runB, registry)` (over t0 cohort; one-side death = disagreement); `lineage_agreement(...)`; `dynamics_distance(...)`; `spatial_pattern(state_map)` → patch centroid/area/Rg/front + `iou/ssim(mapA, mapB)`; `field_match(...)` internal check.

- [ ] **Step 1:** Write failing `test_identical_runs_agreement_one` (same run twice → agreement 1.0, IoU 1.0), `test_shifted_pattern_metrics`, `test_raster_common_extent_mismatched_frames` (CPM integer frame vs tyssue continuous → same grid — Review Focus), `test_one_side_death_counts_as_disagreement`.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement (scan-fill polygons; skimage for SSIM).
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 16: Visualizations

**Files:**
- Create: `viva_multicell_blueprint/viz/spatial_playback.py`, `viz/dynamics_overlay.py`, `viz/divergence.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Produces: `spatial_playback(runCPM, runVertex)` (two synchronized animated panels, shared time slider), `dynamics_overlay(runs, bands)` (CPM vs vertex trajectories + Sego band ribbons), `divergence(runs, registry)` (per-cell agreement curve + spatial divergence map). All return Plotly figures, frames decimated ≤28, coords rounded, `include_plotlyjs="cdn"`.

- [ ] **Step 1:** Write failing `test_playback_has_two_synced_panels`, `test_dynamics_overlay_has_band_ribbons`, `test_figure_under_inline_cap` (serialized HTML < `_HTML_PER_FILE_MAX`).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement (registered v2 visualizations).
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 17: The Sego 2022 blueprint (expressiveness acceptance)

**Files:**
- Create: `workspace/blueprints/influenza_sego2022.yaml`
- Test: `tests/test_sego_blueprint.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the full Sego model as one blueprint — 4 fields, H/I/D transitions + resistance + allee + infected death as `rate_expr` rules, M/NK/CD8 immune component, Price-2015 ODE component.

- [ ] **Step 1:** Write failing `test_sego_blueprint_validates`, `test_sego_compiles_to_both_backends`, `test_expresses_every_sego_rate` (assert each Sego rate — infection, resistance, infected death, allee death/recovery, NK kill, CD8 kill, IL-10 secretion — is present as a blueprint rule, none hidden in code). This is the template's acceptance test.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Author the blueprint, filling parameters from `viva_cpm_studies/influenza/params.yaml` with provenance.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Commit.

---

## Task 18: Studies (investigation `influenza-crossparadigm`)

**Files:**
- Create: `workspace/investigations/influenza-crossparadigm/investigation.yaml` + `studies/{adhesion-calibration,initial-state-fidelity,flu-lung-crossparadigm,dose-response-crossparadigm}/` with `study.yaml` + `tests/`
- Test: each study's `tests/` (behavior tests via the workbench `run` fixture)

**Interfaces:**
- Consumes: the Sego blueprint + compare + viz.
- Produces: side-by-side runs + comparison + figures per study; rigor spine (controls, seeds for the stochastic CPM side, falsifiability, honest open questions). Capstone `flu-lung-crossparadigm` reports per-cell agreement, dynamics-with-bands, and spatial-pattern metrics.

- [ ] **Step 1 (adhesion-calibration):** sweep neutral strength → match mechanics-only shape statistics across backends; record the mapping constants. Behavior test: both backends reach comparable cell-shape stats.
- [ ] **Step 2 (initial-state-fidelity):** both backends reproduce t=0 centroids/sizes/types; report the vertex 1:1 residual. Test: residual below threshold.
- [ ] **Step 3 (flu-lung-crossparadigm capstone):** run the Sego blueprint on both backends at a CI-reduced scale; compute all three comparison families + render the viz. Tests: runs complete; agreement/dynamics/pattern metrics computed; figures render under the cap.
- [ ] **Step 4 (dose-response, stretch):** do both backends agree on the Fig-5/7 monotone direction.
- [ ] **Step 5:** `vivarium-workbench gen-readme --check` + `lint-workspace.py` green; `/viva-report --audit` clean. Commit.

---

## Notes on execution order & parallelism

- **Task 9 (viva-cpm PR)** is an upstream prerequisite for Task 10; start it early (its own repo/worktree) so review/merge overlaps other work.
- **Tasks 2-8** (blueprint + shared mechanism, tested via FakeBackend) are independent of the real backends and of each other beyond Task 3→7 and 5→{6,7,8} — parallelizable.
- **Tasks 10, 11** (backends) depend on Task 5; **13** depends on 10/11/12; **14/15/16** depend on the backends; **17/18** depend on everything.
