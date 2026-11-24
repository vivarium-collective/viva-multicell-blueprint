# Model-definition format — Milestone 1 (format core + compile) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A declarative `scene + registry + bindings + adapter choices` that `compile()`s into a runnable process-bigraph composite, proven by running a small mixed scene whose cells use three different intracellular formalisms — SBML-ODE, Smoldyn particles, and a MaBoSS Boolean network — coupled to a shared extracellular field.

**Architecture:** A thin **compiler/assembler** over existing process-bigraph engine wrappers. The format carries *semantics* (a scene + per-type model specs from a registry); `compile()` chooses *mechanics* (an intracellular adapter per cell + a tissue backend + an extracellular representation) and wires one sub-process per cell to the mechanics backend and a shared field through a declared membrane contract. No new simulators — adapters wrap `viva-tellurium` (ODE), `viva-smoldyn` (particles), `viva-maboss` (Boolean), with the `viva-pde-particle` seam reserved for per-molecule particle species.

**Tech Stack:** Python 3.12, process-bigraph; this repo's vertex/CPM/munk backends + bounded-particles coupling; `viva-tellurium`, `viva-maboss`, `viva-smoldyn` (engine wrappers, reached in-process or via the `SmoldynRunner`-style bridge where a binary/env mismatch requires it); numpy for the baseline extracellular field.

**Spec:** `docs/superpowers/specs/2026-11-23-model-definition-format-design.md` — read it; this plan implements its "Resolved decisions" slice and defers its "Open questions".

## Global Constraints

- **Reuse, don't reinvent.** Intracellular/extracellular behaviour comes from existing wrappers (`viva-tellurium`, `viva-maboss`, `viva-smoldyn`, `viva-pde-particle`). A new numerical simulator in this milestone is a scope error.
- **Keep today's blueprint alongside.** Do not modify or reframe `fields`/`rules`/`systemic_ode`/`subcellular`. The new format is additive, in a new `viva_multicell_blueprint/model_format/` package, and shares the process-bigraph compile target.
- **Binding context = cell type only** this milestone; keep the binding key a struct so richer context can be added without a schema change.
- **Extracellular default = field**; a molecule becomes particles only when declared, via the `viva-pde-particle` seam.
- **Engine optionality.** `viva-tellurium`/`viva-maboss` are optional extras; a missing engine degrades gracefully (adapter reports unavailable; the composite still runs the other cells) — never an import-time crash.
- **No AI attribution** in commits/PRs. Conventional-commit titles. Never auto-merge.
- **Env parity with the bridge pattern.** If an engine lacks a 3.12 binary in this venv, wrap it with the same in-process-or-subprocess `Runner` pattern already used for Smoldyn (`viva_multicell_blueprint/particles/runner.py`); do not force a venv rebuild.

## Review Focus

- **Rate reconciliation:** two adapters (ODE vs particle) for the *same* model spec must reach the same equilibrium only after the documented rate translation; a test pins agreement within tolerance, and an un-translated control shows disagreement. (Task 4/8.)
- **Unbound cell type:** a scene cell whose `type` has no registry entry must fail at compile with a clear error, not at run time. (Task 1.)
- **Orphan / missing species:** a scene `molecule_levels` entry for a species no bound model declares, and a model species absent from the scene (defaults to 0) — both handled explicitly, not a KeyError. (Task 2/7.)
- **Declared-crossing with no extracellular home:** a species marked membrane-permeable but with no extracellular representation configured must error at compile. (Task 6.)
- **Engine unavailable:** `viva-maboss`/`viva-tellurium` not importable — the adapter reports unavailable and the compile either errors clearly (if a bound cell needs it) or proceeds (if none do). (Task 3/5.)

---

### Task 1: Scene + registry + binding data model

**Files:**
- Create: `viva_multicell_blueprint/model_format/scene.py`, `registry.py`, `binding.py`
- Test: `tests/model_format/test_scene_registry_binding.py`

**Interfaces:**
- Produces: `Scene` (`cells: dict[str, CellSpec]`, `domain`), `CellSpec` (`type: str`, `position: [x,y]`, `molecule_levels: dict[str,float]`), `ModelSpec` (`formalism: str`, `source: str|dict`, `params: dict`, `membrane: dict[str, Transport]`), `Registry.get(type) -> ModelSpec`, `bind(scene, registry) -> dict[cell_id, ModelSpec]`.

- [ ] Write failing tests: a 3-cell scene (types `A`,`B`,`C`) + a registry with those three `ModelSpec`s resolves to a `{cell_id: ModelSpec}` map; an unbound type raises `UnboundCellType`; a scene round-trips through YAML.
- [ ] Run tests — expect failure (module missing).
- [ ] Implement the dataclasses + `Registry` (dict-backed, keyed by `type`) + `bind()` + YAML (de)serialization.
- [ ] Run tests — expect pass. Commit.

### Task 2: The intracellular ModelAdapter interface

**Files:**
- Create: `viva_multicell_blueprint/model_format/adapters/base.py`
- Test: `tests/model_format/test_adapter_base.py`

**Interfaces:**
- Produces: `IntracellularAdapter` ABC — `__init__(model_spec, initial_levels)`, `step(dt, inputs: dict) -> None`, `amounts() -> dict[str,float]`, `crossed() -> dict[str,float]` (species that left via the membrane this step), `available() -> bool`. `inputs` carries extracellular concentrations at the cell + mechanical state. A `NullAdapter` reference implementation (holds amounts) proves the contract.

- [ ] Write failing test: a `NullAdapter` constructed from a `ModelSpec` returns its initial `amounts()`, `step` is a no-op, `crossed()` is empty.
- [ ] Run — fail. Implement the ABC + `NullAdapter`. Run — pass. Commit.

### Task 3: ODE adapter (SBML via viva-tellurium)

**Files:**
- Create: `viva_multicell_blueprint/model_format/adapters/ode_tellurium.py`
- Test: `tests/model_format/test_ode_adapter.py`

**Interfaces:**
- Consumes: Task 2 `IntracellularAdapter`. Produces: `ODEAdapter(IntracellularAdapter)` wrapping `viva-tellurium`/libRoadRunner; `ModelSpec.source` is SBML (string/path) or a reaction-network dict it loads.

- [ ] Write failing test: a 1-reaction `A+B<->C` SBML model, seeded, reaches its analytic equilibrium under `step()` within tolerance; if `viva-tellurium` is unimportable the test skips and `available()` is False.
- [ ] Run — fail. Implement using `viva-tellurium` (guard the import; `available()` reflects it). Run — pass/skip. Commit.

### Task 4: Particle adapter (Smoldyn) + the rate contract

**Files:**
- Create: `viva_multicell_blueprint/model_format/adapters/particle_smoldyn.py`
- Create: `viva_multicell_blueprint/model_format/rates.py`
- Test: `tests/model_format/test_particle_adapter.py`, `tests/model_format/test_rate_reconciliation.py`

**Interfaces:**
- Consumes: Task 2 interface; this repo's `particles.runner.SmoldynRunner` + `smoldyn_cell`. Produces: `ParticleAdapter(IntracellularAdapter)` confined to the cell's footprint polygon; `rates.to_particle(spec)` / `rates.to_ode(spec)` translating one canonical rate constant to each engine's convention (the calibration from the intracellular spike).

- [ ] Write failing tests: (a) particle adapter conserves non-reacting counts and forms `C`; (b) **rate reconciliation** — ODEAdapter and ParticleAdapter built from the SAME `ModelSpec` agree on equilibrium `C` within tolerance *after* `rates` translation, and a control without translation does not.
- [ ] Run — fail. Implement the adapter (reusing `SmoldynRunner`) + `rates` translation; document the mapping. Run — pass. Commit.

### Task 5: Boolean adapter (MaBoSS via viva-maboss) — the MaBoSS cell

**Files:**
- Create: `viva_multicell_blueprint/model_format/adapters/boolean_maboss.py`
- Test: `tests/model_format/test_boolean_adapter.py`

**Interfaces:**
- Consumes: Task 2 interface; `viva-maboss`. Produces: `BooleanAdapter(IntracellularAdapter)` — `amounts()` returns node activities (0/1 or probabilities); `inputs` can pin input nodes (e.g. a receptor-bound fraction from the field); exposes a decision output (e.g. a `fate` node) for the up-coupling.

- [ ] Write failing test: a tiny 3-node Boolean network (input → intermediate → output) settles to the expected attractor under `step()`, and pinning the input node flips the output; skips + `available()` False if `viva-maboss` is unimportable.
- [ ] Run — fail. Implement via `viva-maboss` (follow `viva-multiscale-nfkb`'s usage; guard the import). Run — pass/skip. Commit.

### Task 6: Membrane contract + extracellular field

**Files:**
- Create: `viva_multicell_blueprint/model_format/membrane.py`, `extracellular.py`
- Test: `tests/model_format/test_membrane_extracellular.py`

**Interfaces:**
- Consumes: `ModelSpec.membrane` (`{species: Transport(kind=reflect|cross|transporter, rate)}`). Produces: `FieldExtracellular` (numpy diffusing grid; `deposit(species, pos, amount)`, `concentration_at(pos, species)`, `diffuse()`), and a `membrane.apply(adapter_crossed, cell_pos, field)` that routes a cell's crossed molecules into the field and returns per-cell uptake inputs. A species declared `cross` with no extracellular representation raises at build.

- [ ] Write failing tests: a sender cell's crossed species accumulates in the field and a neighbour's `concentration_at` rises; a `cross` species with no field configured raises `NoExtracellularHome`. (Particle-species extracellular is interface-stubbed with a `TODO: viva-pde-particle` and a skipped test.)
- [ ] Run — fail. Implement `FieldExtracellular` + `membrane.apply`. Run — pass. Commit.

### Task 7: compile(scene, registry, bindings, config) → process-bigraph composite

**Files:**
- Create: `viva_multicell_blueprint/model_format/compile.py`
- Test: `tests/model_format/test_compile.py`

**Interfaces:**
- Consumes: Tasks 1–6. Produces: `compile_scene(scene, registry, *, mechanics, extracellular, intracellular_overrides) -> dict` — a process-bigraph composite document: one cell sub-process per cell (adapter chosen by `ModelSpec.formalism`, overridable per-type), wired to the tissue mechanics backend (this repo's `_compile(..., backend=)`) and the shared extracellular store via the membrane contract. Missing engines → the config errors if a bound cell needs them.

- [ ] Write failing test: compiling a 2-cell scene (one ODE type, one particle type) returns a document that `Composite(...)` instantiates and `run(3)`s without error; the cells' adapters are the declared formalisms; an unbound type errors at compile.
- [ ] Run — fail. Implement the assembler (reuse the bounded-particles process shape as the per-cell particle sub-process; a lightweight step-process for ODE/Boolean cells). Run — pass. Commit.

### Task 8: `@composite_generator` + mixed-formalism demo scene

**Files:**
- Create: `viva_multicell_blueprint/composites/scene_model.py`
- Create: `workspace/scenes/mixed_formalism_demo.yaml`
- Test: `tests/model_format/test_scene_model_composite.py`

**Interfaces:**
- Consumes: Task 7. Produces: a registered `scene_model` generator (params: `scene`, `mechanics`, overlay_out) running a demo scene with **three cell types — SBML-ODE, particle, MaBoSS** — sharing one field, over ~16 steps.

- [ ] Write failing test: the generator registers; the composite runs end-to-end; each cell's adapter matches its type's formalism; `n_cells`/per-formalism outputs are emitted. (MaBoSS/tellurium cells skip-degrade if their engine is unavailable.)
- [ ] Run — fail. Implement the generator + the demo scene YAML + `register_workspace_processes` wiring. Run — pass. Commit.

### Task 9: Demo flipbook + investigation study

**Files:**
- Create: `viva_multicell_blueprint/model_format/viz.py`
- Create: `workspace/investigations/particle-multicell-composition/` study `scene-model-mixed-formalism/study.yaml`

**Interfaces:**
- Consumes: Task 8 run output. Produces: a cell-coloured flipbook (cells by type/formalism, the field as a heatmap, particle cells showing molecules) written to a study `viz/` artifact, matching the repo's existing flipbook style.

- [ ] Write failing test (smoke): rendering the demo run produces a self-contained HTML under the inline size cap with ≥1 frame.
- [ ] Run — fail. Implement the viz (reuse the bounded-particles render) + the study YAML. Run — pass. Commit.

---

## Self-review notes

- **Spec coverage:** scene/registry/binding (T1), the three adapter axes + the formalisms incl. the MaBoSS cell (T2–T6), the membrane + extracellular-field decision (T6), compile→composite (T7), a runnable mixed-formalism demo + viz (T8–T9). Deferred by design: imaging→scene ingestion, full FEniCSx PDE + per-molecule particle extracellular (interface-stubbed in T6), automated rate calibration (manual translation in T4), richer binding context.
- **Interfaces consistent:** `IntracellularAdapter` (T2) is implemented identically by T3/T4/T5 and consumed by T7; `rates` (T4) is the single translation point; `FieldExtracellular` (T6) is the one extracellular store T7 wires.
- **Right-sizing:** each task ends at an independently testable deliverable; T3/T5 (engine-dependent) isolate the optional dependencies so a reviewer can accept the core (T1–T2, T4, T6–T7) even if an engine is missing in CI.
