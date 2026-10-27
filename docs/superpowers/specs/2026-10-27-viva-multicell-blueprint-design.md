# viva-multicell-blueprint — Design

**Status:** draft for approval · 2026-10-27
**Owner:** eranagmon
**Destined home:** `viva-multicell-blueprint/docs/superpowers/specs/2026-10-27-viva-multicell-blueprint-design.md` (moves into the repo when it is scaffolded; lives in the session scratchpad until approval, since building — including scaffolding — waits on sign-off).

---

## 1. Intent — the why

Every multi-cell tissue model is written against **one** simulation paradigm. The Sego 2022 influenza-lung model is a Cellular Potts Model (CPM); a vertex modeler writes the "same" tissue a completely different way. When such a model predicts an emergent behavior — an infection front spreading at some speed, a dead patch of a certain shape — you cannot tell whether that behavior is **biology** or an **artifact of the mechanical substrate** (a pixel lattice with stochastic flips vs. a graph of vertices relaxing under tension).

`viva-multicell-blueprint` is a **cross-paradigm reproducibility instrument**. You describe a multi-cell model **once**, in backend-neutral terms (a *blueprint*), and the tool compiles and runs it on **both** a CPM backend (`viva-cpm`) and a vertex backend (`vivarium-tyssue`) from **identical initial conditions and identical rules**, then compares the two runs on both **disease-progression dynamics** and **emergent spatial pattern**.

The payoff:
- Where CPM and vertex **agree**, the result is substrate-independent → trustworthy biology.
- Where they **diverge**, you have learned exactly which predictions depend on how you chose to represent cell mechanics — itself a scientific finding, never a bug to paper over.

The first model carried is the Sego 2022 influenza model, **extracted from `viva-cpm` and generalized** so its mechanism no longer assumes a CPM substrate.

## 2. Goals

1. **A process-bigraph blueprint type** — a registered, validated typed document (`multicell_blueprint`) that specifies a multi-cell model in backend-neutral terms: domain, cell types, initial geometry, type-pair adhesion, per-type mechanics, reaction-diffusion fields, typed rules, immune agents, and systemic-ODE coupling.
2. **A compiler** that lowers one blueprint onto each backend: `compile(blueprint, backend="cpm"|"vertex") -> composite_document`. The compiled composites share **identical** mechanism-process stores and differ **only** in the single mechanics-process store.
3. **A shared mechanism runtime** — backend-neutral process-bigraph processes for the fields, the cell-state rules, the immune agents, and the systemic ODE. One code path drives both runs. (User decision: *shared engine, only mechanics differ*.)
4. **A mechanics-backend interface** — a common process contract that both the CPM and the vertex engines satisfy: it consumes per-cell fate/parameter updates plus the field (gradient) each cell is exposed to, and emits per-cell `{position, size, type, neighbors}` each step.
5. **A data-loading path** — fill a blueprint's initial state from experimental data (a segmentation **label image** or a **cell table**), so different datasets run through the same pipeline unchanged.
6. **Side-by-side studies** that run the full flu-lung blueprint on both backends and **compare**: per-cell state agreement (cells keep a shared identity across backends), dynamics trajectories with the paper's acceptance bands overlaid, and emergent spatial pattern rasterized onto one common grid.
7. **Rich, interactive visualizations** in each study: synchronized side-by-side spatial playback of both substrates, dynamics overlays with band ribbons, and divergence maps.

## 3. Non-goals (v1)

- Not a general-purpose replacement for either backend's native capabilities. The shared engine exists to make the *comparison* fair; `viva-cpm` and `vivarium-tyssue` keep their own native machinery for their own work.
- Not a third mechanical backend. CPM + vertex only; the interface is designed so a third (e.g. center-based / Delaunay) can be added later without touching the mechanism or the blueprint.
- Not substrate-identical state agreement. The comparison lives at shared observables + rasterized pattern, never at "same pixel / same vertex."
- Not a visual blueprint editor. Blueprints are YAML + the data loader.

## 4. Anti-goals (a run that violates these has failed even if it "works")

- **No tuning to force agreement.** No per-backend fudge constant introduced solely to make CPM and vertex line up. A divergence is reported honestly.
- **No second, secretly-different field engine.** The entire premise is **one** authoritative field/rule engine feeding both. If the vertex run quietly diffuses virus differently from the CPM run, the instrument is void.
- **No mechanism invented beyond what Sego 2022 / `viva-cpm` provides.** Parameters and rules come from the extracted model; anything new is recorded as a `pending` proposal, never silently used (ecosystem house rule).

## 5. The central architectural move

`viva-cpm` already splits the influenza model into a **scene** (a declarative CPM spec: geometry + fields + chemotaxis wiring, consumed by `load_world`) and a **mechanism** (the per-MCS rule drivers in `run.py` / the `Infection/Immune/SystemicODE` processes that the studies actually measure). We **generalize that split**:

- The **mechanism becomes backend-neutral shared processes** in the new repo.
- The **scene becomes two compile targets** (CPM spec vs. tyssue mesh+effectors).
- A thin **MechanicsBackend** port is the only seam between them.

```
                 ┌──────────────────────────────────────────────┐
   blueprint ──▶ │  compile(blueprint, backend)                  │
   (one doc)     │    → shared mechanism stores (IDENTICAL)      │
                 │    → one MechanicsBackend store (DIFFERS)     │
                 └──────────────────────────────────────────────┘
                                    │
      ┌─────────────────────────────┼──────────────────────────────┐
      ▼                             ▼                                ▼
 FieldProcess            RulesProcess / ImmuneProcess          MechanicsBackend
 (RD on shared grid)     / SystemicODEProcess                  = CPMBackend  OR  VertexBackend
 virus/IFN/chemo/IL-10   infection, resistance, death,         places cells at blueprint
 secretion+uptake at     killing, allee, recovery,             centroids+sizes; evolves
 cell centroids          recruitment, Price-2015 ODE           shape/position; reports
      │                             │                          per-cell {pos,size,type,nbrs}
      └──────────── shared stores (fields, fates, counts, ode_state, immune_agents) ─────────┘
```

**Why fields move out of CPM.** CPM diffuses/secretes fields natively; tyssue has no fields at all. To keep **one** authoritative field engine (user decision), the shared `FieldProcess` owns all reaction-diffusion on a fixed Eulerian grid, with secretion sources and uptake placed at **cell centroids by type**. Each backend then only **reads** the field to drive motility:

- **CPM backend:** native diffusion/secretion are disabled (`d=0`, `decay=0`, no secretion); the shared field grid is written into the engine each step via a **new bulk `World.set_field(grid)` method added to viva-cpm** (the engine today exposes only a per-voxel additive write, which is too slow at paper scale — see §6 cross-repo change), so CPM's native chemotaxis biases its Metropolis flips off the *authoritative* field. CPM contributes its stochastic flip mechanics; the field is not re-diffused.
- **Vertex backend:** the field **gradient** is sampled at each cell centroid and applied as cell motility via the fork's `ActiveMigration` effector (`vivatyssue` `src/tyssue/dynamics/effectors.py:699`), whose per-cell migration director (`mx/my/mz`, `migration_strength`) is **rewritten every step from the gradient through the existing `behaviors` port** — no change to `EulerSolver`. Division / apoptosis / differentiation also enter through that `behaviors` port.

**Immune cells are substrate cells, not a shared off-lattice layer.** In Sego, M/NK/CD8 are real cells placed on the lattice; their chemotaxis is substrate motility and their killing is **contact-surface-area** based. So in this design they are **cell types materialized on each backend** (CPM cells / tyssue faces). Their *movement* is therefore part of mechanics and legitimately differs between backends — it is one of the things the comparison measures. What stays shared, bit-for-bit, is everything chemical and rule-based: field dynamics, secretion, the infection/resistance/death/killing/recovery **rate rules**, recruitment, and the systemic ODE. A kill rule reads the per-cell **shared-surface-area-by-type** the MechanicsBackend emits (CPM contact area; tyssue shared edge length) — the *rule* is written once; the *quantity* it reads is backend-native.

So the independent variable is exactly: **how cells (epithelial and immune) move and deform.** Chemistry, infection, immunity-as-rules, and the systemic ODE are identical across the two runs.

## 6. Components & file structure

New repo `viva-multicell-blueprint`, a process-bigraph workspace (scaffolded via `/viva-workspace` from `viva-template`). Import package `viva_multicell_blueprint`. Depends (via `[tool.uv.sources]` git) on `viva-cpm`, `vivarium-tyssue`, `vivarium-workbench`, `process-bigraph`.

```
viva_multicell_blueprint/
  blueprint/
    schema.py          # the multicell_blueprint type + validation
    model.py           # Blueprint dataclass / loader (yaml ⇄ object)
    data_loading.py    # from_label_image(), from_cell_table() → initial_state
  compile/
    base.py            # MechanicsBackend contract (ports) + registry
    to_cpm.py          # blueprint → viva-cpm spec + CPMBackend store
    to_vertex.py       # blueprint → tyssue mesh/effectors + VertexBackend store
    adhesion.py        # type-pair adhesion → CPM J-matrix / tyssue tension (documented map)
  mechanism/           # SHARED, backend-neutral (lifted+generalized from viva-cpm)
    fields.py          # FieldProcess: RD on the shared grid, centroid secretion/uptake
    rules.py           # RulesProcess: transitions, killing, allee, resistance, recovery
    immune.py          # immune RULES only (secretion, contact-kill rate, recruitment);
                       #   immune CELLS are materialized on the substrate by each backend
    systemic_ode.py    # SystemicODEProcess: Price-2015 (reused ~as-is; no engine imports)
  backends/
    cpm_backend.py     # wraps viva_cpm CPMProcess behind MechanicsBackend
    vertex_backend.py  # wraps vivarium_tyssue EulerSolver behind MechanicsBackend
  compare/
    identity.py        # shared cell-id ⇄ (cpm cell id, tyssue face id) registry
    raster.py          # rasterize either backend's cell-state map onto the common grid
    metrics.py         # per-cell agreement, dynamics distance, spatial pattern metrics
  viz/
    spatial_playback.py  # synchronized CPM|vertex animated state maps (Plotly)
    dynamics_overlay.py  # CPM vs vertex trajectories + band ribbons
    divergence.py        # per-cell agreement over time + spatial divergence map
workspace/
  blueprints/influenza_sego2022.yaml   # the extracted, generalized flu-lung blueprint
  investigations/influenza-crossparadigm/
    studies/...        # see §9
```

**Extraction policy (viva-cpm → here).** Import the **pure-math helpers** from `viva_cpm_studies.influenza` where they are already substrate-independent (`price_ode`, and the rate math in `transitions/killing/allee/resistance/fields` — confirmed zero engine imports), so we do **not** fork the science. Reimplement only the piece that is CPM-coupled (the `run.py` per-MCS driver, which carries ~181 engine references and reads CPM voxels) as the backend-neutral `RulesProcess`/`FieldProcess` reading per-cell state from the MechanicsBackend.

**One cross-repo change to viva-cpm (approved).** Add a small, additive, backward-compatible **`World.set_field(name_or_idx, grid)`** bulk setter to the Rust `World` + its Python binding (today only a per-voxel additive `field_add_source_at` exists, and `step()` always re-diffuses). This lets the shared `FieldProcess` write the authoritative field into CPM's grid each step (with native dynamics zeroed) for CPM chemotaxis. It changes no existing behavior and ships as its own viva-cpm PR. `vivarium-tyssue` needs **no** core change.

### MechanicsBackend port (the only seam between shared mechanism and substrate)

```
MechanicsBackend  (both CPMBackend and VertexBackend satisfy this process contract)
  config:  { initial_state, cell_types, adhesion (lowered), domain, backend params }
  inputs:  fates            : map[cell_id -> new_type]        # from RulesProcess
           field_grid       : tree  (name -> grid)           # authoritative, from FieldProcess
           divisions/deaths : list  (lineage events)         # from RulesProcess
  outputs: positions        : map[cell_id -> [x,y(,z)]]      # centroid
           sizes            : map[cell_id -> float]          # volume (CPM) / area (vertex)
           types            : map[cell_id -> type]
           shared_surface_by_type : map[cell_id -> map[type -> float]]
                                                             # CPM contact area; vertex shared edge length
           neighbors        : map[cell_id -> list[cell_id]]
           alive            : map[cell_id -> bool]
```

The shared `RulesProcess` consumes `positions/sizes/types/shared_surface_by_type/neighbors` + the field, evaluates every `rate_expr`, and emits `fates` + lineage events. This port is what makes "write each rule once" true.

## 7. The blueprint type (backend-neutral document)

**Expressiveness model — how the *full* Sego model is specified through the template (not hard-coded).** The template is expressive along three registers, and translation-to-two-frameworks happens **only at the mechanics seam**, never per-rule — so adding mechanism never means writing CPM code and tyssue code twice:

1. **Structural-declarative** — domain, cell types, adhesion, fields (D/decay/secretion/uptake/chemotaxis), initial state, immune agent types, ODE coupling map. Pure data; the compiler reads it directly.
2. **Bounded rate grammar** — each `rule` carries a `rate_expr` written in a small, closed expression language over a fixed variable set, chosen so every Sego rate is expressible and nothing wider is:
   - a cell's own `type` and `size` (the kill rate divides by cell volume/area);
   - its exposure to any named `field` (mean over the cell / at centroid);
   - its `neighbors` — counts by type **and** `shared_surface[type]` (the contact-area-by-type the port emits; the Sego contact-kill term needs this, not just a neighbor list);
   - **per-cell scalars produced by earlier rules in the same step** (e.g. `resist` from the resistance rule feeds the death and kill rates) — rules form a small ordered dependency chain, each writing a named per-cell scalar the next may read;
   - named blueprint constants;
   - functions `exp`, Hill `x/(k+x)`, `min`/`max`, and arithmetic.

   It is deliberately **not** a general programming language. The compiler parses each `rate_expr` once and evaluates the identical AST on both backends against the per-cell state the MechanicsBackend emits — so a rule is authored once and runs the same in CPM and vertex.
3. **Parameterized named components** — the genuinely complex, non-expressible pieces (the Price-2015 **systemic ODE**; the off-lattice **immune-agent** motility/killing/recruitment model) are **registered mechanism components** that the blueprint *selects and parameterizes* (`systemic_ode: {model_ref: price2015, coupling:{...}}`), not re-derives. They live in the shared `mechanism/` package, so both backends get the identical implementation; the blueprint supplies their constants and wiring.

**The full Sego 2022 model ships as the canonical reference blueprint** (`workspace/blueprints/influenza_sego2022.yaml`): all 4 fields, H/I/D epithelial transitions + resistance + allee + death as `rate_expr` rules, M/NK/CD8 as parameterized immune components, and the Price-2015 ODE as a named component. It is both the v1 target **and** the proof that the template is expressive enough — if it can't say the whole Sego model, the template is wrong. Generality for *other* models then comes from: different structural data, different `rate_expr` rules, a different (or absent) ODE/immune component, and data-filled initial state.

Sections of the document:

- `domain`: `{extent_um:[x,y(,z)], grid_resolution_um, boundary:"noflux"|"periodic"}` — defines the shared field grid.
- `cell_types`: list of `{name, target_size, mechanics:{...neutral mechanical knobs...}, mobile:bool}`.
- `adhesion`: type-pair interaction `[{a, b, strength}]` in **neutral units**, plus a documented lowering to each backend (see §8).
- `initial_state`: either **generative** `{kind:"blocks"|"voronoi", ...}` or **from_data** `{label_image: path}` / `{cell_table: path}`. Produces the canonical cell list `[{id, type, centroid, size, footprint?}]`.
- `fields`: `[{name, D, decay, secretion:[{type, rate}], uptake:[{type, rate}], chemotaxis:[{type, strength}], dynamics:{dt, substeps}}]`.
- `rules`: typed list — `transition {from, to, rate_expr}` (rate over field exposure / neighbors / type), `kill {killer, target, rate}`, `allee {...}`, `resistance {...}`, `recovery {...}`.
- `immune`: agent types + recruitment coupling to the ODE.
- `systemic_ode`: `{model_ref, coupling:{...}}`.
- `mechanics_mapping`: optional per-backend overrides + the adhesion-translation parameters.

`Blueprint.compile(backend)` returns a composite document whose mechanism stores are identical across backends and whose single mechanics store is backend-specific.

## 8. Initial-state mapping & shared identity (the crux of "same cells both sides")

The canonical cell list `[{id, type, centroid, size, footprint?}]` is produced once (generative or from data). Each backend materializes it, and an **identity registry** (`compare/identity.py`) records `blueprint_cell_id ⇄ cpm_cell_id` and `blueprint_cell_id ⇄ tyssue_face_id`:

- **CPM:** seed each cell as a region at its centroid with `target_volume = size` (via `seed_block`, or `seed_labels` directly when `initial_state` is a label image — `viva-cpm` already supports a label array).
- **Vertex:** build a tyssue sheet from a **domain-clipped Voronoi tessellation of the centroids**, with per-face `cell_type` and `prefered_area = size`; write the mesh to the `.hf5` the `EulerSolver` loads. The fork's mesh builders (`from_2d_voronoi` etc., exercised in `vivarium_tyssue/hra/loaders.py`) **do not guarantee one face per input centroid** — they add a boundary guard collar and weld coincident vertices. So the registry is built by **nearest-centroid assignment** over interior faces, **guard/collar faces are tagged and excluded** from the per-cell comparison, and the residual (centroids that didn't get a clean 1:1 interior face) is **measured, not assumed zero** — that is exactly what the `initial-state-fidelity` study (§9.2) reports. CPM seeding is exact 1:1 (`seed_from_labels`, `viva_cpm/schema.py:47-56`); the vertex side is 1:1 in the interior with a quantified boundary residual.

**Adhesion lowering** (`compile/adhesion.py`): the neutral type-pair `strength` maps to CPM's contact energy `J(a,b)` directly, and to tyssue **line tension on edges whose two incident faces are of types (a,b)**. These are different energy forms, so the mapping is documented and its calibration is itself a study (§9.1) — not hidden.

**Cells born/lost mid-run — explicit scoring rule** (division, CPM fragmentation, tyssue T1/T2):
- The **primary per-cell agreement metric is computed over the cohort of cells present at t=0 in both backends** (the clean shared-id interior cohort). Each such cell's state is compared at every checkpoint.
- A cohort cell that **dies/extrudes on one side but is alive on the other counts as a disagreement** from the time of divergence (alive vs dead is a real state difference, not a bookkeeping gap).
- **Division-born cells** (no t=0 shared id) are tracked by **lineage** (parent id) and scored in a **separate lineage-level metric** — "do the same regions proliferate?" — never folded into the primary per-cell score.
- A cell that **vanishes from one backend's bookkeeping entirely** (CPM fragmentation / tyssue T2 removal with no death event) is flagged as a mechanics artifact and excluded from the per-cell score but **counted and reported** (its rate is itself a backend-divergence finding).

## 9. Studies (investigation `influenza-crossparadigm`)

1. **`adhesion-calibration`** — calibrate the neutral adhesion → (J, tension) mapping so a *mechanics-only* sheet (no infection) reaches comparable cell-shape statistics on both backends. Documents the mapping; produces the numbers the later studies use.
2. **`initial-state-fidelity`** — verify both backends reproduce the blueprint's initial centroids/sizes/type layout within tolerance, and **report the vertex-side 1:1 residual** (fraction of blueprint cells without a clean interior face, boundary collar excluded). Establishes that the shared identity holds at t=0 and quantifies where it doesn't.
3. **`flu-lung-crossparadigm`** (capstone) — run the full generalized Sego blueprint on both backends, identical shared mechanism, and compare:
   - **dynamics:** `counts{H,I,D,M,NK,CD8}` vs t, field integrals vs t, Price-2015 ODE states vs t — CPM vs vertex overlaid, with Sego Fig-3B/5/7 bands as reference.
   - **per-cell agreement:** fraction of shared-identity cells in the same state at each checkpoint; time-to-infection correlation; H/I/D confusion.
   - **spatial pattern:** rasterize each run's cell-state map onto the common grid; compare dead/infected **patch centroid, area, radius-of-gyration, front position**, and an IoU/SSIM of the state maps; compare field maps (should match closely — an internal check that the shared engine really is shared).
4. (Stretch) **`dose-response-crossparadigm`** — do both backends agree on the Fig-5/Fig-7 monotone dose-response direction?

Each study follows the ecosystem rigor spine (controls, seeds/replicates for the stochastic CPM side, falsifiability, honest open questions).

## 10. Visualizations (interactive Plotly, per study)

- **Synchronized spatial playback:** two panels, CPM voxel map | vertex polygon map, cells colored by state (H/I/D/immune), field as a heat underlay, one shared time slider; decimated frames + rounded coordinates to stay within the workbench's per-figure inline cap.
- **Dynamics overlay:** CPM vs vertex trajectories per observable, band ribbons behind.
- **Divergence panel:** per-cell agreement curve over time + a spatial map at each checkpoint showing *where* the two backends disagree.

## 11. Data loading ("easily filled by experimental data")

`data_loading.from_label_image(path)` and `from_cell_table(path)` both emit the canonical cell list. A label image → CPM `seed_labels` directly and → vertex Voronoi from label centroids. A cell table (`id, x, y, type, size`) → both the same way. Swapping datasets is: point `initial_state.from_data` at a new file; nothing else changes.

## 12. What's left to the builder (bounded choices)

- Exact adhesion J↔tension calibration constants — fixed by the `adhesion-calibration` study; v1 may start from a documented direct formula and refine.
- Shared grid resolution vs. the vertex length scale — pick so field numerics stay stable (`substeps*dt == 1` MCS, as in `viva-cpm`'s `fields.py`) and the raster is faithful; constraint, not a free choice.
- Vertex-polygon rasterization (scan-fill faces onto the grid) — standard; builder picks the library call.
- The viva-cpm `World.set_field` Rust signature/layout (grid ordering, field addressing) — builder matches the existing `field_conc`/`field_add_source_at` conventions; the constraint (single authoritative `FieldProcess`, native dynamics zeroed) is fixed.
- The vertex `ActiveMigration` director-update cadence/scaling (how the sampled gradient maps to `migration_strength` + `mx/my/mz`) — fixed by the `adhesion-calibration`/fidelity studies; the mechanism (rewrite directors via the `behaviors` port each step) is fixed.

## 13. Review Focus (inputs the design implies but no obvious test exercises)

1. **Cell disappearing in one backend only** (CPM fragmentation / tyssue T1-T2 face removal) — the identity registry must not mis-align ids; comparison reports it, doesn't crash.
2. **Cells born during the run** (division) — have no t=0 shared id; the per-cell agreement metric must define how it scores them.
3. **Degenerate initial data** — 1 cell, or all one type, or a label image with holes.
4. **Field numerical blow-up** when grid `dt/substeps` aren't stability-matched to a loaded dataset's resolution.
5. **Non-matching domains** — the two backends must rasterize onto the *same* grid extent even if their native coordinate frames differ (CPM integer lattice vs. tyssue continuous positions).
