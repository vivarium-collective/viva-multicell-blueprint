# Blueprint pipeline composite + v2 format + pluggable subcellular modules

**Status:** design, awaiting approval
**Date:** 2026-11-22
**Workspace:** `viva-multicell-blueprint`
**Branch:** `feat/blueprint-pipeline-composite`

## 1. Intent and why

The Blueprint Studio goes from one backend-neutral *blueprint* to a running
multi-cell simulation on any of three tissue engines (Cellular-Potts via
`viva-cpm`, vertex/tyssue via `vivarium-tyssue`, center-based via `viva-munk`).
Today that blueprint→translate→run path is partly hand-written Python, the
blueprint format is a flat schema tied to the influenza use case, and all cell
dynamics are either tissue-global or stateless per-step rules — there is **no way
to run a real subcellular model (an ODE, a boolean network, a dFBA problem, a
stochastic sim) inside each cell**.

We want three things:

1. **Make the pipeline itself a first-class vivarium composite** — a single
   Registry-visible composite built from modular, individually-registered
   Steps/Processes, flexibly configured by the blueprint. The Studio calls *this*
   composite directly. A user can open it in the Registry and examine every piece.
2. **Generalize the blueprint format (v2)** so it cleanly separates *initial
   conditions* from *model definition* and is adaptable across use cases, not
   specific to the flu model.
3. **Add a pluggable per-cell subcellular layer**: the model definition can attach
   a small module (ODE / boolean / dFBA / stochastic) to a cell type; the same
   module runs identically regardless of which tissue engine is selected, and can
   drive the cell's growth, division, death, differentiation, secretion, motility,
   and adhesion. We reuse the existing ecosystem wrappers — `viva-tellurium` (ODE),
   `viva-cobra` (dFBA), and (later) a new MaBoSS wrapper — rather than
   re-implementing solvers.

### Why now
The Studio and the three backends work and are already process-bigraph Processes.
The missing generality is the blueprint format and a subcellular layer. Doing the
format/pipeline refactor and the subcellular seam together — with one module wired
end to end as proof — de-risks the larger ambition (all four module types) before
we invest in each solver.

## 2. Goals / non-goals / anti-goals

### Goals
- A single registered composite generator (working name **`blueprint_pipeline`**)
  that the Studio calls, replacing the direct call to `sego_crossparadigm`.
- The composite is assembled from **modular builder units**, one per concern
  (mechanics backend, fields, tissue rules, systemic ODE, immune, **subcellular**,
  **merge/arbitration**), each selected and configured declaratively by the
  blueprint's `model_definition`.
- Every runtime unit is a **registered Process/Step** visible and individually
  inspectable in the Registry.
- A **v2 blueprint schema** with top-level `initial_conditions` and
  `model_definition` sections, general enough for ODE/boolean/dFBA/stochastic
  subcellular models, with a **v1→v2 loader shim** so existing blueprints and the
  `influenza-crossparadigm` investigation keep running unchanged.
- A **sharded subcellular Process** (one Process per cell-type × module holding all
  that type's cells' solver states keyed by `cell_id`) with a **backend-uniform
  coupling contract**, covering the effect set grow / divide / die / differentiate
  / secrete / **motility** / **adhesion**, plus per-cell readouts.
- **An extended `MechanicsBackend` contract** that gives the subcellular layer real
  actuators across all three backends (see §4f): new per-cell runtime input ports
  (`target_size`, `motility`, `adhesion`) and a division output that surfaces
  **daughter cell_ids + daughter→mother lineage**.
- **Pass 1 proof:** `viva-tellurium` ODE running in cells across all three
  backends, driving per-cell **growth and division** (plus readouts), from a v2
  blueprint.

### Non-goals (this pass)
- Building the MaBoSS wrapper, the dFBA (`viva-cobra`) wiring, or a stochastic
  module. These are Pass 2+; the design must leave clean seams for them but we do
  not build them now.
- Replacing the three backends' native engines or their core mechanics — we *extend*
  their port contract (§4f), we do not rewrite their physics.
- Multi-backend "run all three at once" as a single composite — the Studio keeps
  running the composite once per selected engine (see §4e).

### Anti-goals (would make this a failure even if it "works")
- A v2 format that breaks the existing influenza blueprints or the published
  investigation report. Back-compat via the shim is mandatory.
- A subcellular layer that is secretly backend-specific — if CPM and vertex need
  different wiring for the same ODE, the abstraction has failed. The new per-cell
  ports must mean the same thing on all three backends.
- Per-cell Process *instances* (N processes for N cells). We use the sharded model.
- A "pipeline composite" that is still a monolith — if the mechanics/fields/rules/
  subcellular builders can't be swapped or configured independently, we haven't met
  the "modular steps plugged together" bar.

## 3. Current architecture (grounded)

(From a full read of the package; file:line anchors in the exploration notes.)

- **Run is already process-bigraph.** `CPMBackend`, `VertexBackend`, `MunkBackend`
  subclass a shared `MechanicsBackend(Process)` ABC (`compile/base.py`) with
  identical ports — in: `fates, field_grid, divisions, deaths, activations`; out:
  `positions, sizes, types, shared_surface_by_type, neighbors, alive, footprints`
  (all per-`cell_id` maps). Each drives its native engine inside `update()`.
- **Shared mechanism is registered Processes/Steps:** `FieldProcess` (reaction-
  diffusion), `RulesProcess` (per-cell stochastic fate from a `rate_expr` grammar),
  `SystemicODEProcess` (one *global* ODE), `CellCountStep` (spatial→ODE bridge),
  `ImmuneRecruitmentProcess`. `build_core()` auto-registers every Process/Step
  defined in the package as `local:<ClassName>` and chains imported modules' cores.
- **Translate is imperative Python.** `compile.compile(blueprint, backend)` →
  `to_{cpm,vertex,munk}.compile` → `_shared.assemble(...)` returns a bigraph
  document `{"state": {...}}` of node specs. No Step is involved in building it.
- **The catalog entry** `sego_crossparadigm` is a `@composite_generator` that just
  calls `compile()`. It is already Registry-visible; the Studio page posts to
  `/api/composite-test-run` with this composite id plus `overrides.blueprint` and
  `overrides.backend`.
- **Blueprint format (v1):** flat dataclass / JSON-Schema
  (`.pbg/schemas/multicell_blueprint.schema.json`), required `domain`,
  `cell_types`, `initial_state`; optional `adhesion, fields, rules, immune,
  systemic_ode, mechanics_mapping`. `initial_state` is the only initial-conditions
  field; everything else is global model/dynamics. **No per-cell subcellular model
  exists.**
- **The subcellular gap:** nothing holds per-cell internal state across steps.
  `RulesProcess` recomputes each cell from scratch each tick; the only stateful
  integrator is the single global ODE.
- **Constraints the backends impose today (verified in the builder-check):**
  - The `divisions` event carries only the **mother** id (`{"cell_id": c}`,
    `rules.py:135`, `base.py:92`); no backend emits daughter ids, and `_emit`
    iterates a fixed id set that *excludes* daughters (`munk_backend.py:521`,
    `cpm_backend.py:26`). → division lineage must be **added** (§4f).
  - `target_size` is set once at init (`cpm_backend.py:250`); there is no runtime
    per-cell growth input. Secretion is per-*type* `FieldProcess` config, not a
    per-cell port. → growth/secretion/motility/adhesion actuators must be **added**
    (§4f).
  - `RulesProcess` writes `fates`/`scalar` as `overwrite` stores (`rules.py:82-87`).
    A second writer to the same store is a conflicting update, not a merge. → the
    subcellular layer writes **separate** stores, merged by an arbitration Step
    (§4c).
  - `Blueprint.from_dict` validates against the v1-only schema and rejects unknown
    top-level keys (`model.py:43-45`). → the shim must run **before** validation and
    the internal object stays **flat** (§4a).

### Reuse availability
- `viva-tellurium` — real pbg `TelluriumProcess(Process)` (ODE/SBML/Antimony via
  RoadRunner), config `model`/`model_format`/`integrator`/overrides; has a
  `coupled_species` operator-split mode. **Installable; not yet in this venv.**
- `viva-cobra` — `FBAStep` + `MichaelisMentenUptakeProcess`, dFBA composite, and a
  `ShardedDFBA` "many cells share one model" pattern. Not in the catalog; install
  from `~/code/viva-cobra`. Pass 2.
- `viva-simbio` — CRN/ODE alternative to tellurium. Optional.
- **MaBoSS — does not exist anywhere in the ecosystem.** Must be built from scratch
  (via `/viva-expert`) in Pass 2+.

## 4. Design

### 4a. Blueprint v2 format

Two top-level sections. `initial_conditions` is "where the cells start and what
they physically are"; `model_definition` is "what dynamics run." Field *contents*
are reused from v1 unchanged wherever possible — the change is grouping plus the
new `subcellular` block.

```yaml
blueprint_version: "2.0"

initial_conditions:
  domain:        { extent_um, grid_resolution_um, boundary }
  cell_types:                       # taxonomy + physical identity only
    - { name, target_size, mechanics?, mobile? }
  adhesion:      [ { a, b, strength }, ... ]
  initial_state: { kind: blocks|voronoi, ... }   # or { from_data: {...} }

model_definition:
  fields:    [ ... ]        # reaction-diffusion (v1 `fields`, unchanged)
  rules:     [ ... ]        # tissue-level stochastic fate grammar (v1 `rules`)
  immune:    { ... }        # v1 `immune`
  systemic:  { model_ref, params, coupling }   # v1 `systemic_ode`, renamed
  subcellular:              # NEW — per-cell-type pluggable modules
    <cell_type_name>:
      module: tellurium | maboss | cobra | stochastic
      model:  <inline antimony/SBML/ref>        # module-specific
      params: { ... }
      coupling:
        inputs:  { <model_var>: <tissue_source>, ... }
        outputs: { <tissue_effect>: <model_var_or_expr>, ... }
  mechanics_mapping: { ... }   # v1, backend/run knobs
```

- **`cell_types` lives in `initial_conditions`**; the subcellular *dynamics* attach
  by cell-type *name* from `model_definition.subcellular`, so the two sections stay
  decoupled and either can be edited alone.
- **Generality:** `subcellular.<type>.module` names a registered subcellular
  backend; `model` is whatever that backend consumes. The `coupling` block is the
  single general contract (§4c) every module type obeys.
- The authoritative schema `.pbg/schemas/multicell_blueprint.schema.json` is bumped
  to accept **both** shapes: v2 (`initial_conditions`+`model_definition`) and the
  legacy flat v1 layout, `additionalProperties:false` within each.

**v1→v2 shim (ordering is load-bearing).** `Blueprint.from_dict` / `Blueprint.load`
**first** detect the document shape (presence of `initial_conditions`/
`model_definition`, or `blueprint_version` `< 2`), **then** normalize, **then**
validate, **then** construct. The internal `Blueprint` representation stays the
**existing flat dataclass** plus one new field `subcellular: dict` — i.e. a v2 doc
is *flattened* (`initial_conditions.*` and `model_definition.*` hoisted to the flat
fields; `systemic`→`systemic_ode`) so every `_shared.py` / `sego.py` reader
(`blueprint.domain`, `.systemic_ode`, `.mechanics_mapping`, …) keeps working
unchanged. A v1 doc is simply already-flat. Result: the four existing
`workspace/blueprints/*.yaml` and the influenza studies run untouched. A
`multicell-blueprint migrate-blueprints` CLI (dry-run first) can rewrite files to
v2 on disk, but is not required for correctness.

### 4b. The pipeline composite

A single registered generator, working name **`blueprint_pipeline`**
(`composites/pipeline.py`), `@composite_generator(..., core_extensions=
[register_workspace_processes])`. Parameters: `blueprint` (path / inline YAML /
dict, v1 or v2), `backend` (`cpm|vertex|munk`), `video_out`, plus run knobs.

It replaces the monolithic `compile()` with **per-concern builder units** — small,
independently-testable functions emitting the node spec(s) for one concern:

| Builder unit | Emits (registered node) | Selected/configured by |
|---|---|---|
| `build_mechanics(backend, ic, mapping)` | `local:{CPM,Vertex,Munk}Backend` | `backend` param + `initial_conditions` |
| `build_fields(model_def)` | `local:FieldProcess` | `model_definition.fields` |
| `build_rules(model_def)` | `local:RulesProcess` | `model_definition.rules` |
| `build_systemic(model_def)` | `local:SystemicODEProcess` + `local:CellCountStep` | `model_definition.systemic` |
| `build_immune(model_def)` | `local:ImmuneRecruitmentProcess` | `model_definition.immune` |
| **`build_subcellular(model_def, ic)`** | **one `local:SubcellularProcess` per cell-type with a `subcellular` entry** | `model_definition.subcellular` |
| **`build_merge(...)`** | **`local:CellActuationMergeStep`** (arbitration, §4c) | present iff subcellular present |
| `build_video(...)` | `local:SpatialVideoStep` | `video_out` |

The generator composes only the builders whose config is present — the
`model_definition` is the plug board. `backend` swaps exactly one builder's output
(the mechanics node); everything else is identical across engines. Adhesion lowering
(`adhesion.py`) folds into `build_mechanics`. The old `compile/*` and `sego.py` stay
as thin shims delegating to the new builders (so `sego_crossparadigm` and
`compare/runner.py` keep working through Pass 1), retired in a later cleanup.

**Registry visibility.** `blueprint_pipeline` registers as a composite generator;
its constituents are `local:*` Processes/Steps, so the Composite Explorer shows the
full wiring and each node is individually runnable.

### 4c. The sharded subcellular layer + arbitration

`SubcellularProcess(Process)` in `mechanism/subcellular.py`, registered
`local:SubcellularProcess`. One instance per (cell-type × module). It holds a dict
`cell_id → solver-state` and advances all of that type's live cells each update.

- **config:** `cell_type` (str), `module` (str), `model` (payload), `params` (map),
  `coupling` (inputs/outputs maps), `division` (`copy|split|reset`, default
  `copy`), `seed`.
- **inputs (per-`cell_id` maps — the backend output ports + new lineage):** `types`,
  `sizes`, `positions`, `neighbors`, `alive`, `field_grid`, plus the `divisions`
  event list **now carrying daughter ids + lineage** (§4f) and `deaths`.
- **outputs (written to SEPARATE subcellular stores, never the Rules stores):**
  `sub_scalar` (per-cell internal readouts), and per-cell *actuation requests*:
  `req_target_size`, `req_divide`, `req_die`, `req_become`, `req_secrete`,
  `req_motility`, `req_adhesion` — each a per-`cell_id` map.
- **Field sampling:** the Process samples `field_grid` (a `map[node]` input) at a
  cell's centroid directly, reproducing the nearest-site idiom used privately in
  `RulesProcess._exposure` / `FieldProcess._site` (there is no shared *value*
  sampler today — `fields.field_gradient_at` returns a gradient, not a value — so
  this idiom is lifted into a small shared helper).
- **Lifecycle (buildable now that §4f adds lineage):** on a `divisions` event the
  daughter ids are known; the daughter's state is derived from the mother per the
  `division` policy (`copy` inherit / `split` halve extensive quantities / `reset`
  re-init); on `deaths`/`alive=false`, drop the state.

**`CellActuationMergeStep`** (`mechanism/actuation.py`, registered
`local:CellActuationMergeStep`) resolves the subcellular layer's actuation requests
(and, where they overlap, the tissue `rules` outputs) into the single authoritative
per-cell channels the mechanics backend consumes:
- `req_become` + Rules `fates` → merged `fates` (documented precedence: an explicit
  subcellular differentiation wins over a stochastic rule, or vice-versa — a config
  `arbitration: subcellular_first|rules_first`, default `rules_first` to preserve
  current influenza behavior).
- `req_divide`/`req_die` → `divisions`/`deaths` event lists (unioned with Rules').
- `req_target_size`/`req_motility`/`req_adhesion` → the new per-cell mechanics input
  ports (§4f). `req_secrete` → a per-cell secretion channel merged into
  `FieldProcess` input.
This Step is the single place overwrite-store collisions are avoided: Rules and
Subcellular each write their own stores; only the merge writes the backend inputs.

**The coupling contract (general, module-agnostic):**
- `coupling.inputs` map a *model variable* → a *tissue source* from a fixed
  vocabulary: `field.<name>@centroid`, `size`, `neighbors.count`,
  `neighbors.shared_surface.<type>`, `type`, `scalar.<name>`, `const.<name>`.
- `coupling.outputs` map a *tissue effect* → a *model variable / threshold
  expression*, from the fixed effect vocabulary: `target_size` (grow), `divide_when`
  (divide), `die_when` (die), `become.<type>` (differentiate), `secrete.<field>`
  (secrete), `motility` (migration bias), `adhesion` / `adhesion.<type>` (contact
  strength). Readouts: any model variable is exposed in `sub_scalar` for plotting.
- This contract is identical for every module type and backend. A module adapter
  only builds N solvers from `model`, steps them with mapped inputs, and reads mapped
  outputs — ODE species, boolean node states, dFBA fluxes, stochastic counts all fit
  "named variables in, named variables out."

### 4d. Tellurium ODE — the Pass 1 proof

- Install `viva-tellurium` into the workspace venv; add it to `workspace.yaml`
  `imports` so `build_core` chains its core.
- `module="tellurium"` adapter wraps RoadRunner (reusing `TelluriumProcess`'s build
  path) in the sharded holder. `model` = Antimony/SBML text; `coupling.inputs` set
  species/params from tissue each step; `coupling.outputs` read species/rates back to
  drive `target_size` (grow) and `divide_when`, with per-cell species as readouts.
- New example blueprint `workspace/blueprints/subcellular_ode_demo.yaml` (v2): one
  cell type carrying a small growth ODE whose output drives size and triggers
  division past a threshold; a diffusing nutrient field feeds the ODE input. Runs on
  `cpm`, `vertex`, and `munk`, producing per-cell internal-species trajectories
  alongside the spatial playback.

### 4e. Studio repoint

- `workspace/reports/assets/blueprint_studio.html`: change the hardcoded composite
  id constant(s) — `COMPOSITE` (and the second `sego_crossparadigm_large` reference
  at ~line 121) — to `...composites.pipeline.blueprint_pipeline`. The Studio's
  existing `overrides.blueprint` / `overrides.backend` contract is unchanged (the new
  generator takes the same params). The "all three engines" option keeps issuing one
  run per engine. (`workbench_viewers.py` only supplies the launcher URL and needs no
  change.)

### 4f. MechanicsBackend contract extension (all three backends)

This is the new backend work the "full" scope requires. It is additive — existing
ports and behavior are unchanged; non-subcellular runs ignore the new ports.

- **New per-cell runtime INPUT ports** on `MechanicsBackend` (and honored in each
  `update()`): `target_size` (`map[cell_id→float]`, overrides the per-cell growth
  target), `motility` (`map[cell_id→float]`, migration bias), `adhesion`
  (`map[cell_id→float]` or `map[cell_id→map[type→float]]`, per-cell contact
  strength). Each backend maps these onto its native knob (CPM target volume /
  temperature-like bias / contact energy; tyssue preferred area / line tension;
  munk target radius / drag / adhesivity). Absent/empty → current init-time behavior.
- **Division OUTPUT lineage:** the `divisions` event and `_emit` cohort are extended
  so a division surfaces the **daughter cell_id(s)** and a `lineage`
  (`map[daughter→mother]`); the daughter enters the emitted per-cell maps the step it
  is born. This touches each backend's division path, including CPM's currently
  untested latent path (`cpm_backend.py:350-354`) — that path gets a focused test as
  part of this work.
- Backend parity is guarded: with no subcellular layer and empty new ports, every
  backend reproduces its current run bit-for-bit (golden tests, §7).

## 5. Staging / plan

**Pass 1 (this effort):**
1. v2 schema (accepts v1+v2) + flat internal `Blueprint` + `subcellular` field +
   shim-before-validate; all existing blueprints/tests green through the shim.
2. `MechanicsBackend` contract extension (§4f) across CPM/vertex/munk: new input
   ports + division lineage; parity goldens; CPM division-path test.
3. `blueprint_pipeline` generator + per-concern builders + `CellActuationMergeStep`;
   `sego.py`/`compile/*` become shims; `sego_crossparadigm` still works.
4. `SubcellularProcess` (sharded) + coupling contract + lifecycle; tellurium adapter;
   `viva-tellurium` installed/imported.
5. v2 `subcellular_ode_demo.yaml`; runs on all three backends with growth + division.
6. Studio repointed to `blueprint_pipeline`.
7. Tests (§7) and a short Registry walkthrough confirming the composite is visible
   and inspectable.

**Pass 2+ (designed-for, not built now):** `viva-cobra` dFBA module (leaning on
`ShardedDFBA`); a new MaBoSS pbg wrapper via `/viva-expert`, then a `maboss` module;
a `stochastic` module. Each is a new module adapter behind the same coupling contract
+ a catalog install — no change to the pipeline, the merge step, or the format.

## 6. Constraints
- Back-compat: `influenza-crossparadigm` studies and the four existing blueprints
  must run unchanged via the shim; the published report must not break; §4f is
  additive and parity-guarded.
- Three backends only; we extend their port contract, not their physics.
- Pass 1 adds exactly one new runtime dependency (`viva-tellurium`); no MaBoSS/cobra
  installs in Pass 1.
- Worktree discipline: all work on `feat/blueprint-pipeline-composite`; tests run
  against the worktree (PYTHONPATH or editable install of the worktree), never the
  shared checkout.

## 7. Testing strategy
- **Shim:** every v1 `workspace/blueprints/*.yaml` loads, normalizes, validates, and
  compiles to the same bigraph document it does today (golden-dict comparison); a v2
  doc flattens to the same internal object.
- **Backend parity (§4f):** with empty new ports and no subcellular layer, each
  backend reproduces its current run (golden); new input ports measurably change the
  native knob; division emits daughter id + lineage on all three (incl. the CPM
  latent path).
- **Pipeline parity:** `blueprint_pipeline` on the influenza blueprint matches
  `sego_crossparadigm` per backend, guarding the refactor.
- **Builders + merge:** each builder emits the expected node from a minimal config;
  absent config → no node; `CellActuationMergeStep` resolves overlapping Rules +
  Subcellular writes per the documented arbitration.
- **Subcellular:** `SubcellularProcess` steps N cells; `copy/split/reset` daughter
  policies on a real division event; death drops state; coupling inputs/outputs
  resolve; the tellurium demo runs on all three backends and per-cell species evolve,
  respond to the mapped field input, and drive growth + division.
- **Studio:** `test_blueprint_studio.py` / `test_studio_video.py` pass against the
  repointed composite id.

## 8. Calls I made (settle unless you object)
- Composite name `blueprint_pipeline`; mechanics-only backend selection (fan-out
  stays in the Studio); keep `rules`, rename `systemic_ode→systemic` (shim maps it).
- Arbitration default `rules_first` (preserves current influenza behavior); daughter
  policy default `copy`.
- A dedicated `CellActuationMergeStep` owns all writes to backend input ports, so
  Rules and Subcellular never collide on an overwrite store.
- Internal `Blueprint` stays flat (+`subcellular`), v2 flattened on load; schema
  accepts both shapes; shim runs before validation.
- Tellurium adapter drives RoadRunner in the sharded holder rather than one
  `TelluriumProcess` instance per cell (performance); contract unaffected.

## 9. What's left to the builder
The builder implements §4a–§4f and §5 Pass 1 per the above, choosing internal helper
shapes, the exact coupling-source/effect grammar (within the §4c vocabulary), each
backend's native mapping of the new ports, and the golden-test fixtures. Pass 2+
modules are out of scope here.
