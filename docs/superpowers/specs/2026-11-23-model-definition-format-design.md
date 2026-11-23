# General model-definition format: semantics separated from mechanics

**Status:** design (brainstorm output, two spikes done — see "What the spikes proved").
**Author:** Eran (with Claude).
**Date:** 2026-11-23.

## Intent (the why)

Today a model is written as a backend-neutral *blueprint* (cells, adhesion, fields,
rules) that `compile()`s onto one mechanics backend (CPM / vertex / munk). That already
separates *what a tissue is* from *how it moves*. This design generalizes that split so
that:

1. **Initial conditions come from data.** A spatial imaging dataset — each cell's
   position, type, and measured molecule levels — *is* the starting state. We ingest a
   scene, we don't hand-author one.
2. **Each cell's behaviour is a pluggable model from a registry**, chosen by cell type
   (and context), and the registry is **multi-formalism**: one cell type may resolve to
   an **SBML** reaction network, another to a **Boolean/MaBoSS** network, another to a
   rule-based model.
3. **How each thing is *simulated* is a separate, swappable choice at every scale** —
   not baked into the model. The same reaction network can run as a well-mixed ODE, a
   Gillespie process, or a spatial particle sim (Smoldyn); the same tissue can run on
   CPM, vertex, or munk.

The organizing idea is that **the semantics/mechanics split is fractal**: it recurs at
every scale. *Semantics* = what the entities are and the processes governing them (a
scene + per-cell chemistry/logic). *Mechanics* = the simulation method at each scale.
`compile(semantics + mechanics choices) → a runnable process-bigraph composite`.

## Goals

- A **scene schema** that spatial-imaging pipelines can target: cells with position,
  type, and per-cell molecule levels (optionally molecule positions).
- A **model registry** ("model DB") of reusable, multi-formalism cell models, keyed by
  cell type + context, with a binding that maps scene cell-types → models.
- **Three orthogonal, swappable adapter axes**, each proven in a spike:
  - **intracellular**: ODE ↔ Gillespie ↔ particle (Smoldyn), over one semantic model.
  - **membrane**: per-species transport — reflect / cross / transporter-reaction.
  - **extracellular**: continuous field ↔ particle space.
  plus the existing **tissue mechanics** axis (CPM / vertex / munk).
- A model that **carries spatial information** — an SBML network augmented with
  diffusion + geometry (SBML L3 Spatial), so it can be realized as particles.
- **Faithfulness**: two adapters for the same semantic model agree (after an explicit
  rate reconciliation — see below).
- Compiles to the existing **process-bigraph composite** substrate (the current
  `blueprint_pipeline` / `bounded_particles` composites are the precedent).

## Non-goals (for the first version)

- A GUI or an imaging-analysis pipeline (we consume its *output*, a scene).
- Every formalism at once — start with SBML + ODE/particle + one Boolean engine.
- Exact cross-adapter quantitative agreement *without* calibration (calibration is in
  scope; a zero-config perfect match is not).

## Anti-goals

- A format that bakes mechanics into the model (defeats the whole separation).
- A format only expressible for well-mixed chemistry (it must carry spatial, so particle
  realizations are first-class, not bolted on).
- A registry that is just a file dump — models must present a **common coupling
  interface** regardless of formalism, or they are not interchangeable.

## What the spikes proved

Two throwaway spikes (in `scripts/spike_membrane_transport.py` and
`scripts/spike_intracellular_adapter.py`) de-risked the two load-bearing seams:

- **Membrane + extracellular (spike 1).** A membrane-permeable species `S` was declared
  to cross (Smoldyn surface action `trans` instead of `reflect`); it left a sender cell,
  entered a shared **extracellular field** adapter (deposit + diffuse), and was taken up
  by a neighbour. Findings: (a) **per-cell-isolated sims survive crossing *via* an
  adapter** — no single global world needed; (b) the membrane spec is a near-trivial
  per-species surface action; (c) **the extracellular adapter *is* the existing `fields`
  layer** — crossing out = a secretion source, uptake = a sink; (d) the adapter carries
  real physics (its diffusion/decay set signalling range).
- **Intracellular (spike 2).** One reaction network (`A+B⇌C`) ran as a well-mixed ODE in
  one cell and as Smoldyn particles in the adjacent cell, from the same initial counts.
  After calibrating the ODE's effective volume they **agreed** (ODE at the deterministic
  equilibrium, particles fluctuating around it). Finding: **one model → two mechanics is
  real, but rate reconciliation is required** — mass-action `kf` ≠ Smoldyn's particle
  rate (binding radius) ≠ Gillespie propensity. This is the key correctness obligation.

## The shape (decided)

```
scene            (from imaging: cells = {position, type, molecule_levels[, positions]})
  └ per cell: a semantic model resolved from the REGISTRY by (type, context)
       │        semantic model = species + reactions/logic + rate constants + compartments
       │                         (SBML [+ Spatial] | Boolean/MaBoSS | rules)
       ├ intracellular adapter : ode | gillespie | particle        (+ rate reconciliation)
       ├ membrane adapter       : per-species {reflect | cross(rate) | transporter(reaction)}
       └ extracellular adapter  : field | particle-space           (== tissue `fields`)
  └ tissue mechanics: cpm | vertex | munk                           (bounds the cells)
compile(scene + registry bindings + adapter choices + mechanics) → process-bigraph composite
```

### The seams (interfaces a builder must define)

1. **Scene schema.** Cells keyed by id: `position`, `type`, `molecule_levels{species→amount}`,
   optional per-molecule `positions`. This is the typed initial state; `molecule_levels`
   seed every scale (ODE amounts / particle counts). An ingestion adapter maps an imaging
   table to this schema.
2. **Registry + binding.** Models stored and keyed; a binding rule resolves each scene
   cell to a model by `(type, context)`. **"Context" must be defined** — candidates:
   spatial region, neighbour-type composition, an experimental condition tag.
3. **Semantic model + its compartments.** Species, reactions/logic, rates, and the
   compartments `cytoplasm` / `membrane` / `extracellular`. SBML (multi-compartment) is
   the carrier for chemistry; a Boolean/rule model presents the same *interface* (named
   state variables + update) even if its internals differ.
4. **The narrow waists (coupling contract).** For each cell model:
   - **down → cell** (what the model reads): local field concentrations at the cell,
     neighbour types/contacts, mechanical state (pressure/area).
   - **cell → up** (what the model exposes): a decision (divide / die / differentiate),
     secretion/uptake fluxes, a motility/adhesion change.
   - **membrane**: the declared permeable species (and transporter reactions) that bind
     the cytoplasm compartment to the extracellular compartment.
   The extracellular compartment **binds to the shared tissue field** (or particle space).
5. **Adapter axes** as the compile choices (the four listed above).
6. **Rate-semantics contract.** The model declares rate constants in one canonical
   convention; each adapter is responsible for translating them (mass-action ↔ particle
   binding-radius ↔ propensity), with a documented calibration. **This is the single most
   important correctness requirement** and the thing most likely to silently go wrong.
7. **Compile target.** A process-bigraph composite. Each cell becomes a sub-process (its
   chosen intracellular adapter) wired to the mechanics backend and the shared
   field/extracellular store through the membrane contract.

## Prior art to mine (and diverge from)

- **MultiCellDS** — a standard whose *digital snapshot* = our scene and *digital cell
  lines* = our registry. Borrow the snapshot/cell-line split.
- **VCell** — one biological model run as ODE / PDE / stochastic / **Smoldyn particles**;
  the canonical "physiology → math → solver" pipeline. Borrow the pipeline; diverge by
  making the *tissue* mechanics swappable and composing via process-bigraph.
- **SBML Level 3 Spatial** — the standard way to attach geometry + diffusion to SBML;
  the natural "add spatial information" carrier.
- **PhysiBoSS** — PhysiCell cells running MaBoSS Boolean networks per cell; proof that a
  non-SBML formalism plugs into a multicellular mechanics. Generalize it.

## Existing ecosystem pieces to reuse (the adapters already exist)

The format is a **compiler/assembler over existing process-bigraph engine wrappers**, not
a set of new simulators. Each adapter axis already has implementations:

| Axis / role | Implementation (process-bigraph wrapper) |
|---|---|
| intracellular — ODE (SBML) | **viva-tellurium** (libRoadRunner) |
| intracellular — particles | **viva-smoldyn** (+ the bounded-particles coupling in this repo) |
| intracellular — Boolean | **viva-maboss** (MaBoSS) |
| extracellular — field / PDE ↔ particle | **viva-pde-particle** (numpy FV / FEniCSx + Smoldyn) |
| extracellular — PDE (3-D FV) | **viva-vcell-fvsolver** (pyvcell) |
| tissue mechanics | **viva-cpm / vivarium-tyssue / viva-munk** (this repo's backends) |

**Direct precedent: `viva-multiscale-nfkb`.** It already composes **MaBoSS Boolean fate
(Calzone 2010) + SBML-ODE signalling (Lipniacki 2004, via viva-tellurium) + a diffusing
TNF field (viva-pde-particle)** into one process-bigraph composite, where the scales "talk
through small, **checked translation steps**" (e.g. receptor binding `c/(c+Kd)` turns a
field concentration into the cell's input signal). That is exactly this design's coupling
contract + rate-reconciliation, realized once. The format generalizes that composite's
hand-wiring into a declarative `scene + registry + adapters → compile`.

## Resolved decisions

- **Binding context = cell type, for now.** A cell resolves to a model by `type` alone.
  The binding key is structured so richer context (region, neighbour composition,
  condition) can be added later without reworking the scene or registry.
- **Extracellular default = continuous field**, with **per-molecule opt-in to a particle
  representation**. A molecule is a field by default; specific molecules can be declared
  as particles, and the **field↔particle coupling follows `viva-pde-particle`** (its
  PDE/particle hybrid seam: deposit particles → field source, sample field → particle
  flux). So one extracellular space can be mixed (some species field, some particle).
- **Keep today's blueprint alongside, for now.** The existing `fields` / `rules` /
  `systemic_ode` / `subcellular` layers stay as-is; the new format is additive and reuses
  the same compile target. Reframing them as cases of the format is deferred, not assumed.

## Open questions (still the human's call, later)

- **Scene/ingestion**: which imaging output format(s)? Do we get molecule *positions* or
  only *levels*? (Positions would seed particle sims directly.)
- **Registry mechanics**: how are models stored and versioned (a BioModels-style DB?).
- **Rate contract**: the canonical rate convention + per-adapter translation, and whether
  calibration is declared or automated — the one correctness obligation that must be made
  explicit, informed by viva-multiscale-nfkb's "checked translation steps".

## Next spikes (if we keep feeling it out before building)

1. **Imaging → scene** ingestion on a real (or synthetic) spatial dataset.
2. **A Boolean/MaBoSS cell** in the same scene as an SBML cell — prove the non-SBML seam.
3. **The registry + binding** resolving `(type, context)` → model for a mixed population.
