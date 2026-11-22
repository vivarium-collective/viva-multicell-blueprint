# Blueprint Pipeline Composite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the blueprint→translate→run pipeline into one Registry-visible vivarium composite of modular builders, add a v2 blueprint format (initial_conditions / model_definition) with a v1→v2 shim, and add a sharded per-cell subcellular layer (tellurium ODE first) that drives growth/division/death/differentiation/secretion/motility/adhesion uniformly across the CPM/vertex/munk backends.

**Architecture:** A new `@composite_generator` (`blueprint_pipeline`) composes per-concern builder units from the blueprint's `model_definition`. A sharded `SubcellularProcess` (one per cell-type × module) holds per-cell solver state keyed by `cell_id` and writes *actuation requests* to its own stores; a `CellActuationMergeStep` resolves those (plus `RulesProcess` outputs) into the authoritative per-cell channels the mechanics backend consumes. The `MechanicsBackend` contract is extended additively with new per-cell input ports (`target_size`, `motility`, `adhesion`) and a division `lineage` output; each backend maps them onto its native knobs. The blueprint loader normalizes v2 docs to the existing flat `Blueprint` dataclass before validation, so all downstream readers are unchanged.

**Tech Stack:** Python 3.12, `process_bigraph`, existing `viva_multicell_blueprint` package, `viva-tellurium` (RoadRunner), pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-blueprint-pipeline-composite-design.md`

## Global Constraints

- Back-compat is mandatory: the four `workspace/blueprints/*.yaml` and the `influenza-crossparadigm` studies must run unchanged via the shim; the published report must not break.
- The `MechanicsBackend` extension (new ports + lineage) is **additive**: with empty new ports and no subcellular layer, every backend reproduces its current run (parity goldens).
- Three backends only (`cpm`, `vertex`, `munk`); extend their port contract, do not rewrite their physics.
- Pass 1 adds exactly one new runtime dependency: `viva-tellurium`. No MaBoSS/cobra installs.
- All work on branch `feat/blueprint-pipeline-composite`; run tests against this worktree (prepend `PYTHONPATH=<worktree>` or editable-install the worktree), never the shared checkout.
- Commit messages: plain, no AI attribution / no `Co-Authored-By` trailer (per workspace rule).
- Internal `Blueprint` representation stays the flat dataclass + one new `subcellular: dict` field; v2 docs are flattened on load.

## Review Focus

- **Ambiguous blueprint shape:** a doc carrying BOTH a top-level v1 key (e.g. `domain`) AND a `model_definition` section. The shim must reject it with a clear error, not silently pick one. → Task 1.
- **Non-subcellular parity:** a normal influenza run with empty new ports must match the current backend output bit-for-bit on all three backends. → Tasks 3–5 (goldens).
- **Dangling subcellular target:** `model_definition.subcellular` keyed by a cell-type name absent from `initial_conditions.cell_types` must error at build time, not no-op at runtime. → Task 9.
- **Bad coupling reference:** `coupling.outputs`/`inputs` naming a model variable the module does not expose must error with the variable name, not silently drop. → Task 6.
- **Division id integrity:** on division, the emitted `lineage` must map each new daughter id to its mother, daughters must appear in the same step's `positions`/`types`/`alive` maps, and ids must not collide with live cells. → Tasks 3–5.

---

### Task 1: Blueprint v2 format + v1→v2 shim

**Files:**
- Modify: `viva_multicell_blueprint/blueprint/model.py` (add `subcellular` field; normalize-before-validate in `from_dict`)
- Modify: `viva_multicell_blueprint/blueprint/schema.py` + `.pbg/schemas/multicell_blueprint.schema.json` (accept v1 flat AND v2 sectioned)
- Create: `viva_multicell_blueprint/blueprint/v2.py` (`normalize_to_flat(doc) -> dict`)
- Test: `tests/test_blueprint_v2.py`

**Interfaces:**
- Produces: `normalize_to_flat(doc: dict) -> dict` — returns a flat v1-shaped dict (keys `domain, cell_types, initial_state, adhesion, fields, rules, immune, systemic_ode, mechanics_mapping, subcellular, blueprint_version`), accepting either a flat v1 doc (returned ~unchanged) or a v2 doc (`initial_conditions.*` + `model_definition.*` hoisted; `model_definition.systemic`→`systemic_ode`). Raises `ValueError` on a mixed doc (a v1 top-level key present alongside `initial_conditions`/`model_definition`).
- `Blueprint` gains `subcellular: dict = field(default_factory=dict)`.
- `Blueprint.from_dict` calls `normalize_to_flat` first, then `validate`, then constructs.

- [ ] **Step 1: Write failing tests** in `tests/test_blueprint_v2.py`:
```python
def test_v1_doc_loads_flat_unchanged():
    bp = Blueprint.from_dict(_minimal_v1_dict())
    assert bp.domain and bp.cell_types and bp.subcellular == {}

def test_v2_doc_flattens_to_same_object():
    assert Blueprint.from_dict(_v2_dict()).to_dict() == Blueprint.from_dict(_v1_equiv_dict()).to_dict()

def test_v2_subcellular_is_captured():
    bp = Blueprint.from_dict(_v2_with_subcellular())
    assert "epithelial" in bp.subcellular and bp.subcellular["epithelial"]["module"] == "tellurium"

def test_mixed_shape_rejected():
    with pytest.raises(ValueError, match="mixed"):
        Blueprint.from_dict({"domain": {...}, "model_definition": {...}})

def test_existing_blueprint_files_still_load():
    for p in Path("workspace/blueprints").glob("*.yaml"):
        Blueprint.load(p)  # must not raise
```
- [ ] **Step 2: Run tests, verify they fail** — `pytest tests/test_blueprint_v2.py -v`
- [ ] **Step 3: Implement** `normalize_to_flat` in `blueprint/v2.py`; add the `subcellular` field to the dataclass; wire `from_dict` to normalize→validate→construct; extend the JSON schema with a top-level `oneOf` (flat v1 | sectioned v2) and a `subcellular` definition. Keep `unknown top-level keys` guard operating on the *flattened* dict.
- [ ] **Step 4: Run tests, verify pass**; also run the full existing suite to confirm no regression — `pytest tests/test_sego_blueprint.py tests/test_blueprint_studio.py -q`
- [ ] **Step 5: Commit** — `git commit -m "feat(blueprint): v2 initial_conditions/model_definition format + v1->v2 shim"`

---

### Task 2: Extend the MechanicsBackend contract (ABC + FakeBackend)

**Files:**
- Modify: `viva_multicell_blueprint/compile/base.py` (new input ports + `lineage` output port; docstring)
- Modify: `viva_multicell_blueprint/backends/fake_backend.py` (honor new ports; emit lineage on division)
- Test: `tests/test_mechanics_contract.py`

**Interfaces:**
- Produces: `INPUT_PORTS` gains `"target_size": "map[float]"`, `"motility": "map[float]"`, `"adhesion": "map[node]"` (per-cell: `cell_id -> float | {type -> float}`). `OUTPUT_PORTS` gains `"lineage": "overwrite[map[string]]"` (daughter_id -> mother_id, empty most steps). Division events on the `divisions` input stay `{"cell_id": mother}`; the backend chooses daughter ids and reports them via `lineage` + the standard output maps.
- Consumes: Task 1 not required.

- [ ] **Step 1: Write failing tests** in `tests/test_mechanics_contract.py`:
```python
def test_new_input_ports_present():
    be = FakeBackend(config=_cfg())
    assert {"target_size","motility","adhesion"} <= set(be.inputs())

def test_lineage_output_present():
    assert "lineage" in FakeBackend(config=_cfg()).outputs()

def test_empty_new_ports_are_noop():
    # a step with empty target_size/motility/adhesion == a step without them
    assert _run_step(FakeBackend, target_size={}) == _run_step(FakeBackend, omit_new=True)

def test_division_emits_daughter_and_lineage():
    out = _run_step(FakeBackend, divisions=[{"cell_id": "3"}])
    assert out["lineage"]  # daughter -> "3"
    d = next(iter(out["lineage"]))
    assert d in out["types"] and out["alive"][d] is True
```
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** the port additions in `base.py`; in `FakeBackend.update`, apply `target_size`/`motility`/`adhesion` to its toy state and, on a `divisions` event, mint a fresh daughter id (`f"{mother}.{k}"`), add it to all output maps, and record `lineage[daughter]=mother`. Empty new-port maps must leave behavior identical.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Commit** — `git commit -m "feat(backend): additive per-cell actuation ports + division lineage on MechanicsBackend"`

---

### Task 3: CPMBackend honors new ports + division lineage

**Files:**
- Modify: `viva_multicell_blueprint/backends/cpm_backend.py`
- Test: `tests/test_cpm_backend_actuation.py`

**Interfaces:**
- Consumes: Task 2 ports.
- Produces: CPM maps `target_size`→per-cell target volume, `motility`→its temperature/bias knob, `adhesion`→per-cell contact energy; division (the latent `cpm_backend.py:350-354` path) assigns the daughter a new id, adds it to the emitted cohort, and fills `lineage`.

- [ ] **Step 1: Write failing tests**: parity (empty new ports reproduce a current golden run), `target_size` changes a cell's emitted size over steps, and a forced division emits a daughter id + `lineage` entry and the daughter appears alive next step. Mark CPM tests to skip cleanly if `viva_cpm` is unavailable (`pytest.importorskip`).
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** in `cpm_backend.py`, reading the new input maps in `update` and extending `_emit`/the division path to surface daughters + lineage. Do not change default (empty-port) behavior.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Commit** — `git commit -m "feat(cpm): honor per-cell actuation ports + emit division lineage"`

---

### Task 4: VertexBackend honors new ports + division lineage

**Files:**
- Modify: `viva_multicell_blueprint/backends/vertex_backend.py`
- Test: `tests/test_vertex_backend_actuation.py`

**Interfaces:**
- Consumes: Task 2 ports. Produces: tyssue mapping — `target_size`→preferred area, `motility`→migration term, `adhesion`→line tension; division surfaces the new face id + lineage.

- [ ] **Step 1: Write failing tests** mirroring Task 3 (parity golden, target_size effect, division lineage), `pytest.importorskip("vivarium_tyssue")`.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** in `vertex_backend.py`.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Commit** — `git commit -m "feat(vertex): honor per-cell actuation ports + emit division lineage"`

---

### Task 5: MunkBackend honors new ports + division lineage

**Files:**
- Modify: `viva_multicell_blueprint/backends/munk_backend.py`
- Test: `tests/test_munk_backend_actuation.py`

**Interfaces:**
- Consumes: Task 2 ports. Produces: munk mapping — `target_size`→target radius, `motility`→drag/forcing, `adhesion`→adhesivity (`munk_backend.py:210 _parse_adhesivity`); division surfaces the daughter particle id + lineage (`munk_backend.py:521` cohort).

- [ ] **Step 1: Write failing tests** mirroring Task 3, `pytest.importorskip("viva_munk")`.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** in `munk_backend.py`.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Commit** — `git commit -m "feat(munk): honor per-cell actuation ports + emit division lineage"`

---

### Task 6: SubcellularProcess (sharded) + module adapter interface

**Files:**
- Create: `viva_multicell_blueprint/mechanism/subcellular.py` (`SubcellularProcess`, `SubcellularModule` protocol, coupling resolver, a built-in `test_linear` module for dependency-free tests)
- Modify: `viva_multicell_blueprint/mechanism/__init__.py` (re-export `SubcellularProcess`)
- Test: `tests/test_subcellular_process.py`

**Interfaces:**
- Produces:
  - `class SubcellularProcess(Process)` — config `{cell_type:str, module:str, model, params:map, coupling:node, division:str="copy", seed:int=0}`; inputs `types, sizes, positions, neighbors, alive, field_grid` (per-cell maps) + `divisions, deaths` (+`lineage` via field_grid's sibling — read lineage from a `lineage: "map[string]"` input); outputs the request stores `sub_scalar: overwrite[map[map[float]]]`, `req_target_size/req_motility: overwrite[map[float]]`, `req_adhesion: overwrite[map[node]]`, `req_divide/req_die: list[node]`, `req_become: overwrite[map[string]]`, `req_secrete: overwrite[map[map[float]]]`.
  - `SubcellularModule` protocol: `init_cell(cell_id, inputs) -> state`, `step(state, mapped_inputs, interval) -> state`, `read(state) -> dict[var->float]`.
  - `resolve_inputs(coupling_inputs, cell_id, port_maps) -> dict` and `resolve_outputs(coupling_outputs, read_vars) -> dict[effect->value]` — the §4c vocabulary; unknown model variable → `ValueError` naming it.
- Consumes: nothing external (Task 8 adds the tellurium module behind the same protocol).

- [ ] **Step 1: Write failing tests** in `tests/test_subcellular_process.py`:
```python
def test_holds_and_steps_per_cell_state():
    # two cells, test_linear module; sub_scalar evolves per cell
def test_division_copy_split_reset():
    # lineage {daughter: mother}; copy inherits, split halves, reset re-inits
def test_death_drops_state():
def test_coupling_inputs_resolve_field_at_centroid():
def test_coupling_outputs_map_to_requests():
    # target_size request produced from a mapped model var
def test_unknown_coupling_var_errors():
    with pytest.raises(ValueError, match="unknown"):
        ...
```
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** `SubcellularProcess` (state dict keyed by cell_id; spawn daughters from `lineage` per `division` policy; drop on death/`alive=False`), the coupling resolver (centroid sampler lifted from the `RulesProcess._exposure` nearest-site idiom into a shared helper), and the `test_linear` module.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Commit** — `git commit -m "feat(subcellular): sharded per-cell SubcellularProcess + coupling contract"`

---

### Task 7: CellActuationMergeStep

**Files:**
- Create: `viva_multicell_blueprint/mechanism/actuation.py` (`CellActuationMergeStep`)
- Modify: `viva_multicell_blueprint/mechanism/__init__.py` (re-export)
- Test: `tests/test_actuation_merge.py`

**Interfaces:**
- Produces: `class CellActuationMergeStep(Step)` — config `{arbitration:str="rules_first"}`; inputs the Rules outputs (`fates: map[string]`, `divisions/deaths: list[node]`) and the Subcellular requests (`req_become, req_target_size, req_motility, req_adhesion, req_secrete, req_divide, req_die`); outputs the authoritative backend-input channels (`fates: overwrite[map[string]]`, `divisions/deaths: list[node]`, `target_size/motility: overwrite[map[float]]`, `adhesion: overwrite[map[node]]`) plus a `secrete: overwrite[map[map[float]]]` feeding FieldProcess.
- Consumes: Task 6 request stores.

- [ ] **Step 1: Write failing tests**: `rules_first` keeps a Rules fate over a conflicting `req_become`; `subcellular_first` inverts it; `divisions`/`deaths` are the union; `req_target_size` passes through to `target_size`.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** the merge Step.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Commit** — `git commit -m "feat(subcellular): CellActuationMergeStep arbitrates rules + subcellular actuation"`

---

### Task 8: Tellurium module adapter + dependency

**Files:**
- Create: `viva_multicell_blueprint/mechanism/subcellular_modules/tellurium_module.py`
- Modify: `workspace.yaml` (add `viva-tellurium` under `imports`)
- Test: `tests/test_subcellular_tellurium.py`

**Interfaces:**
- Produces: a `TelluriumModule` implementing the `SubcellularModule` protocol (Task 6) and registered under the name `"tellurium"` in `SubcellularProcess`'s module lookup; builds a RoadRunner from `model` (Antimony/SBML), steps each cell's instance by `interval`, exposes species/rates to `read()`.
- Consumes: Task 6 protocol + resolver.

- [ ] **Step 1: Install** `viva-tellurium` into the worktree venv and add it to `workspace.yaml` imports (so `build_core` chains it). Verify `import viva_tellurium`.
- [ ] **Step 2: Write failing test** in `tests/test_subcellular_tellurium.py` (`pytest.importorskip("viva_tellurium")`): a 2-cell growth ODE whose species increases each step and whose mapped output yields a per-cell `req_target_size` that grows; a mapped field input changes the rate.
- [ ] **Step 3: Run, verify fail**
- [ ] **Step 4: Implement** `TelluriumModule` (reuse `TelluriumProcess`'s RoadRunner build path; one RoadRunner per cell held in `SubcellularProcess` state, or a reset-and-set shared instance — implementer's perf call; contract unchanged).
- [ ] **Step 5: Run, verify pass**
- [ ] **Step 6: Commit** — `git commit -m "feat(subcellular): tellurium ODE module adapter + viva-tellurium import"`

---

### Task 9: Pipeline builder units

**Files:**
- Create: `viva_multicell_blueprint/compile/builders.py`
- Test: `tests/test_pipeline_builders.py`

**Interfaces:**
- Produces named builders, each returning a node-spec dict (or `None` when its config slice is absent), delegating to the existing `_shared.py` node builders where one exists:
  - `build_mechanics(blueprint, backend) -> dict` (wraps `to_{cpm,vertex,munk}` mechanics node + adhesion lowering)
  - `build_fields(blueprint) -> dict | None`, `build_rules(...) -> dict | None`, `build_systemic(...) -> dict | None`, `build_immune(...) -> dict | None`, `build_video(...) -> dict | None`
  - `build_subcellular(blueprint) -> list[dict]` (one `SubcellularProcess` node per `subcellular` entry; **raises `ValueError`** if a key is not in `blueprint.cell_types`)
  - `build_merge(blueprint) -> dict | None` (a `CellActuationMergeStep` node iff `subcellular` present)
- Consumes: Tasks 6, 7; `_shared` node builders.

- [ ] **Step 1: Write failing tests**: each builder emits the expected address from a minimal slice; absent slice → `None`; `build_subcellular` on a dangling cell-type name raises `ValueError` naming the type; `build_subcellular` on the demo blueprint yields one node per type.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** `builders.py` delegating to `_shared`.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Commit** — `git commit -m "feat(pipeline): per-concern blueprint builder units"`

---

### Task 10: `blueprint_pipeline` composite generator

**Files:**
- Create: `viva_multicell_blueprint/composites/pipeline.py`
- Modify: `viva_multicell_blueprint/composites/__init__.py` (import it so it registers)
- Modify: `viva_multicell_blueprint/compile/_shared.py` (`assemble` grows an optional `extra_nodes`/subcellular+merge wiring param) and `compile/__init__.py` (so `sego_crossparadigm` keeps working via the builders)
- Test: `tests/test_blueprint_pipeline.py`

**Interfaces:**
- Produces: `blueprint_pipeline(core=None, backend="cpm", blueprint=None, video_out="") -> dict` decorated `@composite_generator(name="blueprint_pipeline", core_extensions=[register_workspace_processes], default_n_steps=80, parameters={backend, blueprint, video_out})`. Assembles mechanics + present mechanism builders + (when `subcellular` present) the subcellular nodes and the merge node, wiring subcellular request stores → merge → mechanics input ports.
- Consumes: Task 9 builders.

- [ ] **Step 1: Write failing tests**:
```python
def test_pipeline_parity_with_sego_each_backend():
    for be in ("cpm","vertex","munk"):
        assert _doc(blueprint_pipeline, _influenza, be) == _doc(sego_crossparadigm, _influenza, be)
def test_pipeline_wires_subcellular_and_merge_when_present():
    doc = blueprint_pipeline(backend="cpm", blueprint=_demo_v2)["state"]
    assert any("Subcellular" in n.get("address","") for n in doc.values())
    assert any("CellActuationMerge" in n.get("address","") for n in doc.values())
def test_pipeline_registered_in_core():
    # the generator resolves against build_core() and builds a Composite
```
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** `pipeline.py` + the `assemble` extension; keep `sego_crossparadigm` delegating through the builders (parity test guards this).
- [ ] **Step 4: Run, verify pass** (incl. the full suite: `pytest -q`)
- [ ] **Step 5: Commit** — `git commit -m "feat(pipeline): blueprint_pipeline Registry composite over modular builders"`

---

### Task 11: Studio repoint + v2 demo + end-to-end

**Files:**
- Modify: `workspace/reports/assets/blueprint_studio.html` (repoint the `COMPOSITE` constant(s) to `viva_multicell_blueprint.composites.pipeline.blueprint_pipeline`)
- Create: `workspace/blueprints/subcellular_ode_demo.yaml` (v2; one cell type with a tellurium growth ODE driving `target_size` + `divide_when`, fed by a nutrient field)
- Test: `tests/test_subcellular_demo_e2e.py`; update `tests/test_blueprint_studio.py` if it asserts the composite id
- Test: `tests/test_studio_video.py` must still pass

**Interfaces:**
- Consumes: Task 10 generator; Task 8 tellurium module.

- [ ] **Step 1: Write failing tests**: `test_subcellular_demo_e2e.py` runs `blueprint_pipeline` on `subcellular_ode_demo.yaml` for a few steps on each available backend (`importorskip` each) and asserts per-cell `sub_scalar` species change over time and at least one cell grows (size increases) and a division occurs (a `lineage` entry appears). Assert the Studio html references `blueprint_pipeline`.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** the demo blueprint, repoint the html constant(s), adjust `test_blueprint_studio.py` expectation.
- [ ] **Step 4: Run, verify pass**; run `pytest -q` (full suite) and confirm the influenza/studio/video tests are green.
- [ ] **Step 5: Commit** — `git commit -m "feat(studio): point Blueprint Studio at blueprint_pipeline + v2 subcellular ODE demo"`

---

### Task 12 (optional): `migrate-blueprints` CLI

**Files:**
- Modify: `viva_multicell_blueprint/cli/__main__.py` (add `migrate-blueprints [--dry-run] [paths...]`)
- Test: `tests/test_migrate_blueprints_cli.py`

**Interfaces:**
- Produces: a CLI that rewrites a v1 file to v2 on disk (`--dry-run` prints the diff; default writes), using `normalize_to_flat` inverse (flat→sectioned). Idempotent on v2 input.

- [ ] **Step 1: Write failing test**: round-trip a v1 fixture → v2 file → loads to the same `Blueprint`; `--dry-run` writes nothing.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** the command.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Commit** — `git commit -m "feat(cli): migrate-blueprints v1->v2 rewriter"`

---

## Notes for the executor

- Run the suite against the worktree: `PYTHONPATH=$PWD .venv/bin/pytest -q` (or editable-install the worktree first). The canonical checkout's editable install points elsewhere.
- Tasks 3/4/5 are independent of each other (all depend on Task 2) — safe to parallelize across agents; Tasks 6→7→8 are sequential; Task 9 depends on 6+7; Task 10 on 9; Task 11 on 10+8.
- After Task 11, before any PR: full `pytest -q`, and a manual Registry check that `blueprint_pipeline` renders and its subcellular nodes are inspectable.
