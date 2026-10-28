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
