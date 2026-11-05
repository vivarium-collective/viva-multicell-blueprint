# Response to the vertex-mechanics review (2026-10-28) — accepted, implemented

**Review:** per-study feedback on adhesion-calibration, initial-state-fidelity,
flu-lung-crossparadigm and crossparadigm-equivalence ·
`workspace/references/reviews/influenza_vertex_mechanics_feedback_2026-10-28.yaml`
**Disposition:** all four comments accepted and implemented. The vertex backend was
rebuilt, both substrates' mechanics are now written down, and every vertex-dependent
study was re-run. Design constraint kept: the vertex backend stays ONE standalone,
swappable mechanics node, and all its coupling runs through vivarium-tyssue
*behaviors* passed to its internal `EulerSolver`, the way the Gillespie regulator drives it.

## Point by point

**adhesion-calibration — "lacking neighbor exchanges; rerun with auto_reconnect; define the
mechanical equations and parameters for both backends, including per-cell-type variation."**
- `auto_reconnect` is ON. The T1 merge threshold is 0.1 × the hexagon side (tyssue's
  default of 1e-3 never fires at this cell size). Neighbour exchanges now occur: 4–11 of
  85 epithelial pairs swap by step 400, gains ≈ losses.
- **docs/mechanics.md** defines the CPM Hamiltonian and the vertex energy + dynamics
  term by term. It also lists every parameter per cell type and per type pair, and says
  how each vertex value is set: energy-matched, lowered from the blueprint, or calibrated.
- Defects found and fixed along the way:
  - The compiler lowered adhesion per type pair, but the backend applied only **one
    mean tension** to every edge. A new `junction_tension` behavior now applies the
    full pair table on the live topology every step.
  - Explicit Euler at viscosity 1 was **unstable** for size-25 cells. The
    displacement clamp was hiding a two-state oscillation. Viscosity is now 2·K·A0,
    with 4 sub-steps per coupled step.
  - Vertex cells could leave the domain. The field domain is now a wall, as the lattice
    edge is for the CPM.
- Per-type parameters:
  - Area stiffness is energy-matched per type (K = 2 λ_V).
  - A pair missing from the adhesion table is tension-free on both substrates, since
    viva-cpm's default is J = 0. Previously the vertex used the mean tension.
- New `scripts/calibrate_vertex_mechanics.py` fits the vertex knobs that have no CPM
  counterpart to CPM behaviour (5 seeds × 60 steps), under a **hard mesh-validity
  constraint**: no inverted or self-intersecting face over 3 seeds × 400 steps.

  | Knob | Fitted value | Vertex vs CPM |
  |---|---|---|
  | Tension scale | 1.74 | area 22.8 vs 23.0 |
  | Immune motility | 236.5 | displacement 9.4 vs 9.4 µm |
  | Saturating chemotaxis | — | drift within ~10% for g ≥ 0.003 |

- **Not matched, reported as such:**
  - **Epithelial wandering.** Every sheet motility that matches the CPM's epithelial
    MSD folds the vertex mesh within a few hundred steps. I tried sub-stepping, a
    perimeter stiffness, faster rosette resolution and larger T1 thresholds. The
    motility is therefore capped at the fold-free value, and the vertex epithelium
    wanders ~5× less (MSD 2.0 vs 9.5 µm²).
  - Area CV (0.033 vs 0.093).
  - The tension scale is geometry-dependent (~4 on the 4×4 sheet, which collapses at 8).
- The tension sweep reverses the old finding: tension now **does** set cell area.

**initial-state-fidelity — "hexagonal initial cells; auto_reconnect; one polygon per Sego
cell; split cells sharing a polygon; overlay fields instead of polygons; create the
processes connecting fields to the vertex model via behaviors (Gillespie)."**
- **Hexagonal sheet.** The epithelium is a regular hexagonal tiling at the target
  area. Each blueprint cell gets its own site by an optimal 1:1 assignment, alternated
  with rigid recentring. Coincident centroids are **split** onto neighbouring
  hexagons, never merged.
- **No guard ring.** The 50 guard/medium polygons are gone: 48 faces for 48 cells (36
  hexagons at area 25.0 in one connected sheet, plus 12 immune cells). Residual is
  sub-cell (RMS 1.60 µm, max 2.44 µm).
- **Immune overlay.** Immune cells (and their latent pools) form an **overlay layer**:
  free polygons above the sheet, as in Sego's 2-layer immune geometry. Their contact
  with the epithelium is the overlap area.
- **Fields overlaid.** Fields are sampled onto each face (`field_<name>` columns) by an
  `overlay_fields` behavior and are never meshed.
- **Field coupling via behaviors.** Field → tissue coupling is
  `viva_multicell_blueprint/backends/vertex_coupling.py`. Gillespie-style couplers
  turn fields, fates, latent activations and lineage events into behavior nodes
  (`overlay_fields`, `apply_gradient`, `differentiation`, `split_cells`,
  `junction_tension`, `confine`) on the solver's `behaviors` port.

**flu-lung-crossparadigm — "cells are not actually moving within the tissue; likely the
mechanical parameters and the lack of auto-reconnect."**
- Confirmed, with a third cause: the unstable integration above. Fixed by reconnection,
  stable integration, and CPM-calibrated active motility (the counterpart of the CPM
  temperature).
- Vertex cells now move and exchange neighbours. Immune cells random-walk and chemotax
  like CPM cells. The epithelium is less fluid than the CPM's (see above).
- Re-run against a mechanics-stochastic null control:
  - Fate (0.938 vs 0.926 / 0.944), occupancy IoU (0.135 vs 0.100 / 0.175) and patch
    location (3.6 vs 3.3 µm) are all within seed noise.
  - The old "IoU below both baselines" signal is gone.
  - Remaining candidate: late I-count dynamics. The vertex is inside the narrow CPM
    3-seed envelope at 55% of steps by N=400 (100% at N=150).

**crossparadigm-equivalence — "equivalence is very bad between backends."**
- Two causes found:
  1. Vertex cells did not move.
  2. The null was **degenerate**. Replicates varied only the rule seed, so each
     backend's mechanics were identical across replicates and same-backend positional
     agreement was 1.0 by construction.
- Each backend's mechanics RNG now follows the run seed (`mechanics_seed`).
- Result:
  - Fate is still equivalent (run-cluster CI [−0.003, +0.001] within ±0.05).
  - The exact-position gap fell from **0.92 to 0.135** (cross 0.472 vs same 0.608,
    CI [0.071, 0.210]). That is **still not equivalent** at ±0.10. The remainder tracks
    the less fluid vertex epithelium (replicate spread 2.3 vs 5.2 µm).
- The counterfactual positive control still rejects (0.417 vs 0.775). Its old
  "vertex climbs the gradient" mechanism was an uncalibrated-chemotaxis artifact and
  is retired.

## Still open
- **Epithelial fluidity**: the tyssue sheet folds before it reaches CPM-level epithelial
  wandering. Closing that needs a fold-aware topology handler (or a softer epithelium),
  and it bounds the position-equivalence result.
- Area dispersion: the vertex area CV is lower, and raising it with tension noise folds
  the mesh.
- The tension scale depends on geometry. The calibration holds at one cell size (25).
- The position-equivalence margin needs more seeds, and both margins still need
  biological justification (BP-06).
- Immune contact is edge-on on the CPM but face-on overlap on the vertex, by design,
  so kill rates read different contact units.

---

# Response to the cross-paradigm review — accepted, with revisions

**Reviewer:** Scientific and architecture critique (2026-10-27) ·
`workspace/references/reviews/influenza_blueprint_feedback.yaml`
**Disposition accepted:** `revise` (critical). We agree with the review's central
correction and have reorganized the investigation around it.

## The correction we accept

The investigation was framed around **cross-backend simulation agreement** as if
that were the headline result. That overreached. The real, defensible contribution
is a **declarative, backend-neutral, testable blueprint** with **auditable lowering**
to CPM and vertex substrates. Dynamic equivalence is a *separate* scientific
question, not evidence that the specification is correct. We have re-centered the
investigation on three distinct claims and a five-stage program.

## Three claims, kept distinct (the new spine)

| # | Claim | Question | Status now | Owned by |
|---|---|---|---|---|
| **1** | **Blueprint fidelity** | Can an independently specified biological initial state be represented faithfully in *either* substrate? | **Partial** — centroid/ID/type identity shown; area, footprint, contact, adjacency, boundary, topology, round-trip **not yet** tested. | initial-state-fidelity → Stage 2 (BP-02, BP-03) |
| **2** | **Model portability** | Can the same biological mechanisms connect to each backend **without changing their intended semantics**? | **Partial** — one blueprint compiles to both with a byte-shared mechanism, but V/L/C/F producers are absent and immune recruitment is a latent-pool approximation (named deviations, not "FULL Sego"). | the compiler + blueprint → Stage 1/4 (BP-01, BP-04, BP-05) |
| **3** | **Scientific equivalence** | Under which conditions *should* the two simulations agree, and when do differences reflect meaningful mechanical assumptions? | **Scoped claim** — fate agreement is equivalent within a (not-yet-biologically-justified) margin under a dependence-aware reanalysis; exact position is not; a counterfactual shows the test has power. | crossparadigm-equivalence + substrate-divergence-counterfactual → Stage 3/4 (BP-06, BP-07) |

**We do not infer any claim from another.** Passing the same code on two backends
is not blueprint fidelity and is not biological validity (guardrails below).

## Five-stage work order (dependency-ordered)

Adopted verbatim from the review's recommended order; existing studies and the
reviewer's proposed studies (BP-01…BP-11) are mapped into each stage with an exit
gate. This replaces the previously empty roadmap.

1. **Formalize the blueprint** — canonical biological state + versioned schema,
   separated from backend geometric initialization.
   *Studies:* **BP-01** (schema + canonical IR), **BP-02** (round-trip / metamorphic
   fidelity), **BP-03** (constraint-solver initialization).
   *Exit gate:* canonical schema + portable initial-state fidelity pass on
   pathological and normal fixtures.
2. **Prove initialization fidelity** — beyond centroids/IDs: area, footprint,
   contact, adjacency, boundary, topology, round-trip, explicit error measures.
   *Studies:* **initial-state-fidelity (extend)**, **BP-02**, **BP-04** (contact /
   field-sampling conformance).
   *Exit gate:* geometry/contact/field adapters conform at refinement; losses
   quantified per backend.
3. **Establish scientifically comparable mechanics** — replace the weak adhesion
   mapping with calibration to **common physical observables**; separate geometry,
   initialization, mechanics, and numerical differences.
   *Studies:* **adhesion-calibration (supersede)**, **BP-07** (mechanics
   identification), **BP-10** (variance decomposition).
   *Exit gate:* identifiable (or explicitly unidentifiable) parameters; agreement
   within pre-registered physical-response bands.
4. **Complete + validate the influenza mechanism** — repair V/L/C/F producers,
   resolve recruitment semantics, **reanalyze equivalence (dependence-aware)**,
   compare to literature.
   *Studies:* **BP-05** (complete circuit), **BP-06** (dependency-aware
   equivalence — *started here*), **BP-08** (Sego / lung-epithelial real-data),
   **dose-response (redefine)**, **flu-lung (reconcile)**,
   **substrate-divergence-counterfactual (extend to effect-size sweep)**.
   *Exit gate:* no missing causal inputs; powered equivalence on justified
   observables; measured biology benchmarked.
5. **Demonstrate generality** — compile a **second, independent tissue blueprint**
   through the same schema with no influenza-specific adaptations (the decisive
   generality test), ideally to a third backend.
   *Studies:* **BP-09** (transfer), **BP-11** (scaling).
   *Exit gate:* another blueprint compiles without bespoke influenza branching;
   losses + portability cost quantified.

## What changed in THIS response (now, not promised)

- **Reframed** the investigation's executive around the three claims; demoted
  "cross-backend agreement" from headline to Claim 3.
- **Retired the overstated equivalence statistic.** The 6×6 grid's 36 cross + 30
  same pairs reuse only 6+6 runs and are **dependent**; the pair-level TOST
  (p=5.6e-36) overstated significance. Primary result is now a **run-cluster
  bootstrap**: Δ(same−cross) −0.0015, 90% CI **[−0.003, +0.001]** (still within
  ±0.05). The ±0.05 margin is flagged as **not yet biologically justified** (BP-06).
- **Downgraded confidence** on crossparadigm-equivalence (high→medium) and
  **softened** the spatial finding from "representation artifact" to a *candidate*
  representation effect, **not causally isolated** (BP-02/BP-04 will isolate it).
- **adhesion-calibration:** removed the "comparable / calibrated" wording. A
  23.6 vs 34.9 µm² mean-area gap (~48%) with a tension-insensitive CV is **not** a
  J↔tension calibration; relabeled directional + **non-identifiable** (→ BP-07).
- **flu-lung-crossparadigm:** replaced "FULL Sego" with "Sego model with **named
  deviations**" (absent V/L/C/F producers; latent-pool recruitment approximation);
  stated biological fidelity to Sego figures is **not yet validated** (BP-08).
- **dose-response:** reaffirmed it is **not** a dose–response relationship
  (2 seeds, intermediate CPM seed1 1/3/2 breaks monotonicity; seeded infected
  fraction ≠ virion inoculum) — directional-only (→ BP-05/BP-08, two experiments).
- **Added** the reviewer's **claim guardrails** to the investigation and the
  **figure plan (F1–F10)** as proposed visualizations (F1 "one blueprint, two
  substrates" is P0).
- **Committed** the review itself as a traceable reference artifact.

## Point-by-point

### Investigation sections
- **research-question / abstract / executive-summary** — accepted. Three
  separable claims adopted; blueprint artifact leads; "FULL Sego" removed;
  chronology reconciled (the fail-to-reject reading is **superseded** by the
  dependence-aware equivalence verdict; no study is "refuted" — the status
  contradiction in the published snapshot is corrected in the study specs).
- **blueprint-definition** — accepted → **BP-01** (versioned schema: units,
  coordinate frame, boundaries, tessellation-independent IDs, footprints, fields,
  contacts, latent states, seeds, init order, mechanism bindings, provenance,
  allowable loss; biological-desired-state distinct from realized geometry and
  from dynamic rules).
- **validation** — accepted → **BP-02/BP-04** (compile/inspect/validate/round-trip
  per backend; invariant tolerances + loss reports; expose errors, not a success
  badge; versioning / missing-field / impossible-packing / overlap / tie / ID
  collision / unit / topology / unsupported-feature tests).
- **claim-evidence** — accepted. We now separate **code-level** (compiler runs),
  **numerical** (invariants + stats), and **external biological** (Sego figures /
  measured data, BP-08) evidence, and never cross-infer.
- **synthesis-gaps** — accepted; the dependency-ordered program above replaces the
  empty roadmap.
- **references** — accepted → BP-08 anchors (exact figure/table/parameter IDs,
  extraction, versions, licensing; verbatim vs approximated vs engineered).
- **investigation-visualizations** — accepted → **F1** central "one blueprint, two
  lowerings" figure with transformation + loss reports and common readouts.

### Studies
- **adhesion-calibration** (was "passed") — accepted as **revise**; see above. Methods
  (geometry vs area-elasticity vs tension vs junctions vs boundary guard vs relaxation;
  match physical area first; multi-objective identifiability; many mechanics seeds;
  longer relaxation) and evidence (reference benchmarks, dimensional analysis,
  matched domains, timestep/convergence, boundary-exclusion) → **BP-07**.
- **initial-state-fidelity** (was "passed") — accepted. Centroid+count insufficient;
  per-cell area/perimeter residuals, adjacency, contact-surface, footprint IoU,
  polygon validity, coverage, immutable-ID matching, latent/epithelial separation,
  edge-heavy fixtures, serialize→compile→inspect→reconstruct → **BP-02**.
- **flu-lung-crossparadigm** — accepted. Mechanism-to-producer-to-consumer ledger
  for V/L/C/F + recruitment; instrument every source term; couple exposure to real
  footprints/contact; test update order; ablate frozen inputs; literature
  benchmark → **BP-05/BP-08**. Status reconciled.
- **dose-response-crossparadigm** — accepted; two experiments (infected-fraction
  vs extracellular-virus concentration), 6–10 graded doses, replicate seeds,
  pre-declared hypotheses/error model, dose-vs-fraction plots → **BP-05/BP-08**.
- **crossparadigm-equivalence** — accepted; run/seed as the resampling unit
  (done: cluster bootstrap), fate-agreement denominator + identity matching +
  confusion matrices + per-class hazards, registered scale-aware spatial
  descriptors, pre-registered margins with biological justification → **BP-06**.
- **substrate-divergence-counterfactual** — accepted; factorial sweep over
  adhesion/migration/geometry/chemotaxis/crowding/update-rate on plausible *and*
  extreme ranges; isolate implementation vs representation vs initialization
  inequivalence; add a matched-physical-response control; graded injected
  deviations (power curve) rather than one huge perturbation → **BP-06/BP-07**.

## Guardrails adopted (non-negotiable)

1. Passing the same code on two backends is not evidence of biological validity.
2. Equal total counts can conceal infection-pattern and rare-cell differences.
3. Seed-pair comparisons reuse trajectories — treat statistical dependence
   explicitly (done for equivalence; cluster bootstrap).
4. Failing to reject equivalence is not proof of difference — report bounds + power.
5. CPM contact energy and vertex line tension cannot be equated by a monotonic
   label mapping without shared physical observables.
6. Biological blueprint, compiler mapping, initialized state, dynamic mechanism,
   and readout remain distinct objects.
7. Every declared semantic invariant has a pass/fail test and a quantified
   tolerance.
8. Missing viral/systemic producers and the latent immune pool are **approximations**
   until their causal roles are implemented and validated.
