# Mechanics of the two substrates — equations, parameters, and how they are matched

The cross-paradigm investigation runs ONE blueprint on two mechanical substrates. The
shared mechanism (fields, rules, systemic ODE, recruitment) is byte-identical across
the two compiles; **only the mechanics differ**. This page defines both mechanical
models term by term, lists every parameter (including per-cell-type and per-type-pair
variations) as used for the Sego 2022 influenza blueprint
(`workspace/blueprints/influenza_sego2022.yaml`), and states how each vertex parameter
is obtained from the CPM: energy-matched, lowered from the blueprint, or calibrated.

Units: 1 length unit = 1 lattice site = 1 grid unit (`grid_resolution_um = 1`); area in
sites². One coupled **step** = one shared-mechanism update = `mcs_per_update = 2`
Monte-Carlo sweeps on the CPM and one explicit-Euler step (`dt = 1`) on the vertex model.

## 1. Cellular Potts model (viva-cpm) — `CPMBackend`

Each cell σ is a set of lattice sites with type τ(σ). The effective energy is

$$
H = \sum_{\langle i,j\rangle} J\big(\tau(\sigma_i),\tau(\sigma_j)\big)\,\big(1-\delta_{\sigma_i\sigma_j}\big)
  \;+\; \sum_\sigma \lambda_V(\tau)\,\big(V_\sigma - V_T(\sigma)\big)^2
  \;+\; \sum_\sigma \lambda_S(\tau)\,\big(S_\sigma - S_T(\tau)\big)^2
$$

summed over neighbour pairs ⟨i,j⟩ (neighbour order 2: the 8 nearest sites). Chemotaxis
adds, for a copy attempt from site *s* into *t* by a cell of type τ,
$\Delta H_\text{chem} = -\lambda_\text{chem}(\tau, f)\,\big(c_f(t) - c_f(s)\big)$ on the
shared field *f*. Dynamics: Metropolis copy attempts, accepted with probability
$\min(1, e^{-\Delta H / T})$.

| Symbol | Meaning | Value (Sego blueprint) | Source |
|---|---|---|---|
| J(a,b) | contact energy per neighbour bond | table §3 | blueprint `adhesion` (`J = strength`, exact) |
| J(a,b) for a pair not in the table | — | **0** | viva-cpm default |
| λ_V(τ) | volume constraint | 1.0, all types | CPM default (`cell_types[].mechanics.lambda_volume` overrides) |
| V_T | target volume | 25 sites, all types | blueprint `cell_types[].target_size` |
| λ_S, S_T | surface constraint | 0 (off) | CPM default |
| λ_chem(τ,f) | chemotaxis | M←virus 5000; NK←chemokine 5000; CD8←chemokine 10000 | blueprint `fields[].chemotaxis` |
| T | Metropolis temperature | 10 | CPM default (`potts.temperature`) |
| neighbour order | | 2 | CPM default |
| MCS per step | | 2 | `mechanics_mapping.mcs_per_update` |

Immune cells live in the **same 2-D lattice** as the epithelium (the latent pool is
placed in the medium at the tissue margin), and every cell, immune or epithelial,
wanders under the temperature.

## 2. Vertex model (vivarium-tyssue / tyssue) — `VertexBackend`

**Tissue.** The epithelium (H/I/D) is a confluent sheet of polygons: a **regular
hexagonal tiling** at the target area, one polygon per blueprint cell (optimal 1:1
assignment of blueprint centroids to lattice sites; no guard ring, no polygons for
medium). Immune cells (M/NK/CD8 and their latent pools) are **overlay** cells: free
hexagonal islands above the sheet (Sego's immune layer, z = 1 over the z = 0
epithelium) centred exactly on their blueprint centroid. The shared fields are
**overlaid** on the tissue: sampled at each face centroid into `field_<name>` face
columns, never meshed.

**Energy.** With faces α (area A, perimeter P, type τ) and junctions (half-edges) ij
of length ℓ separating types a, b:

$$
E = \sum_\alpha \frac{K(\tau_\alpha)}{2}\big(A_\alpha - A_0(\alpha)\big)^2
  \;+\; \sum_{\text{junctions}} \Lambda_{ab}\,\ell_{ij}
  \;+\; \sum_\alpha \frac{\Gamma(\tau_\alpha)}{2}\big(P_\alpha - P_0(\tau_\alpha)\big)^2
$$

(tyssue `FaceAreaElasticity` + `LineTension` + `PerimeterElasticity`; the last only if
some Γ > 0). tyssue charges ½Λℓ per *half*-edge, so an interior junction (two
half-edges) carries Λ_ab·ℓ; a border junction (one half-edge, cell vs medium) is set to
2Λ_(a,medium) so it carries the full tension too.

**Dynamics.** Overdamped, explicit Euler, per vertex *i*:

$$
\eta\,\frac{d\mathbf r_i}{dt} = -\nabla_{\mathbf r_i} E \;+\; \sum_{\alpha \ni i} \frac{\mathbf F_\alpha}{n_\alpha}
$$

where each cell's active force $\mathbf F_\alpha$ (tyssue `ActiveMigration`, split over
its $n_\alpha$ vertices) is

$$
\mathbf F_\alpha = F_\text{sat}\tanh\!\Big(\tfrac{|\mathbf C_\alpha|}{F_\text{sat}}\Big)\,\hat{\mathbf C}_\alpha
 \;+\; f_\text{mot}(\text{layer})\,\big(\cos\theta_\alpha, \sin\theta_\alpha\big),
\qquad \mathbf C_\alpha = s_\text{mig}\sum_f \lambda_\text{chem}(\tau_\alpha, f)\,\nabla c_f(\mathbf x_\alpha)
$$

— saturating chemotaxis up the overlaid field gradient plus a persistent random walk
whose polarity diffuses, $\theta_\alpha(t+dt) = \theta_\alpha(t) + \sqrt{2\,dt/\tau_p}\,\xi$.
Junction tension may carry an Ornstein–Uhlenbeck fluctuation
$\Lambda_{ab}(t) = \max\big(0, \Lambda_{ab} + x(t)\big)$,
$x(t+dt) = x\,e^{-dt/\tau_\Lambda} + \sigma_\Lambda\sqrt{1-e^{-2dt/\tau_\Lambda}}\,\xi$, one
draw per physical junction.

**Topology.** `auto_reconnect` is on: a junction shorter than `threshold_length` merges
into a 4-way vertex that detaches along the other axis (T1 neighbour exchange; T3 at the
border), at tyssue's detachment rates (p₄ = 0.1, p₅₊ = 0.01 per unit time). Each coupled
step is integrated in 4 solver sub-steps (`substeps`), so per-update vertex motion stays
well below the T1 threshold. The shared field domain is a hard wall (`confine`
behavior), as the lattice edge is for the CPM.

**Mesh validity is a hard constraint.** An inverted or self-intersecting face is broken
geometry, not a cell shape. Every fitted knob must keep the sheet fold-free over 3 seeds
× 400 steps (§4). `tests/test_vertex_mesh_validity.py` pins this on the Sego run.

**Coupling** to the shared mechanism goes entirely through vivarium-tyssue *behaviors*
handed to the solver's `behaviors` port, the way the Gillespie regulator drives the
`EulerSolver`: `overlay_fields` (field → face columns), `apply_gradient` (migration
director/strength; `is_alive` on death), `differentiation` (rule fates and latent →
active immune activation), `split_cells` (division), and `junction_tension` (per-pair
tension recomputed on the current topology every step, so a junction created by a T1
gets the right tension). See `viva_multicell_blueprint/backends/vertex_coupling.py`.

**Contact surface** (what the rules read as `shared_surface_by_type`): in the sheet, the
shared junction length per neighbour type; for overlay cells, the **polygon overlap
area** with each cell underneath (the 2-layer contact surface), credited both ways.

## 3. Parameters, per type and per type pair (Sego blueprint)

| Parameter | CPM | Vertex | How the vertex value is set |
|---|---|---|---|
| Target size | V_T = 25 | A_0 = 25 | identical (blueprint `target_size`) |
| Size stiffness | λ_V = 1 | K = 2 | **energy-matched**: CPM λ(V−V_T)² ≡ tyssue K/2(A−A_0)² ⇒ K = 2λ_V, per type |
| Surface stiffness | λ_S = 0 | Γ = 0 | energy-matched (Γ = 2λ_S, P_0 = S_T), per type |
| Adhesion | J(a,b) | Λ_ab = κ·J(a,b) | **lowered** per pair; one global scale κ = `tension_scale` **calibrated** (§4) |
| Missing pair | J = 0 | Λ = 0 | identical (viva-cpm default) |
| Fluctuations | T = 10 | f_mot, σ_Λ | **calibrated** (§4) |
| Chemotaxis | λ_chem | s_mig·λ_chem, F_sat | λ_chem ratios identical; s_mig and F_sat **calibrated** (§4) |
| Mobility | (implicit in MCS) | η = 2·K·A_0 = 100 | set for explicit-Euler stability (the area stiffness ~K·A_0 must stay below η/dt) |
| Neighbour exchange | implicit (lattice) | T1 at ℓ < 0.1·side = 0.31 | `threshold_length` scaled to the hexagon side |
| Immune placement | same lattice, at margin | overlay islands at the same centroids | `overlay_types` = immune types + latent pools |

Per-type-pair adhesion (blueprint `strength` = CPM J; vertex Λ = 1.74·J):

| Pair | J | Λ | | Pair | J | Λ |
|---|---|---|---|---|---|---|
| epithelial–epithelial (H/I/D × H/I/D) | 5 | 8.7 | | M–M, NK–NK, CD8–CD8 | 25 | 43.5 |
| epithelial–medium (H/I/D) | 25 | 43.5 | | M–NK, M–CD8 | 10 | 17.4 |
| immune–medium (M/NK/CD8) | 10 | 17.4 | | NK–CD8 | 25 | 43.5 |
| M–H, M–I, M–D | 20 | 34.8 | | NK–I, CD8–I | 10 | 17.4 |
| NK–H, NK–D, CD8–H, CD8–D | 20 | 34.8 | | any latent type × anything | — (0) | 0 |

On the vertex side, immune–epithelium and immune–immune tensions only act where those
cells share a junction. Overlay islands share none with the sheet, so the
epithelium-facing rows are inert there. Contact with the epithelium is the overlap area
instead (§2). Each island's border carries its (type, medium) tension.

Per-type motility layer (vertex): sheet cells (H/I/D) f_mot = 34.1 (fold-free cap, §4);
overlay cells (immune + latent) f_mot = 236.5; persistence τ_p = 10 steps for both.

## 4. Calibration — making the free vertex knobs reproduce CPM behaviour

The temperature T and the vertex mobility/fluctuation knobs share no units, so the
vertex knobs that have no CPM counterpart are **fitted to CPM behaviour** on the same
Sego initial state, mechanics only, averaged over 5 seeds × 60 steps
(`scripts/calibrate_vertex_mechanics.py` →
`workspace/studies/adhesion-calibration/results/vertex_mechanics_calibration.json`;
values in `compile/to_vertex.py: CALIBRATED_DEFAULTS`):

| Knob | Fitted to | CPM | Vertex (fitted) |
|---|---|---|---|
| `tension_scale` κ = 1.74 | mean epithelial area | 23.0 | 22.8 |
| `sheet_motility` = 34.1 (**cap**; MSD fit = 83) | epithelial MSD about the sheet centre, 60 steps | 9.5 µm² | **2.0 µm²** |
| `motility` = 236.5 | median latent-immune displacement, 60 steps (walls on) | 9.4 µm | 9.4 µm |
| `migration_scale` = 3.54, `chemotaxis_saturation` = 105 | macrophage drift speed vs virus gradient g = 0.001 / 0.003 / 0.01 / 0.03 | 0.098 / 0.145 / 0.332 / 0.350 | 0.058 / 0.163 / 0.327 / 0.350 µm/step |
| `tension_noise` = 0 | epithelial area CV | 0.093 | 0.033 (not reachable; see below) |

**What is *not* equivalent, honestly:**

* **Epithelial wandering is ~5× lower.** Matching the CPM's epithelial MSD needs a
  sheet motility of ~83, but at that force (and at any force above ~34) the vertex sheet
  folds within a few hundred steps. Folding persisted with smaller time steps, a
  perimeter (shape) stiffness, faster rosette resolution and larger T1 thresholds.
  tyssue's reconnection resolves short junctions, not a vertex sweeping across a
  non-adjacent edge. So the motility is capped at the largest fold-free value, and the
  vertex epithelium wanders less (MSD 2.0 vs 9.5 µm² over 60 steps). It still exchanges
  neighbours: 4–11 of 85 epithelial pairs swap by step 400, gains ≈ losses (genuine T1s).
* **Area dispersion.** The vertex area CV (0.033) stays below the CPM's (0.093).
  Tension noise raises it, but the level that would match folds the mesh, so it is
  left at 0.
* **Tension scale depends on geometry.** On the isolated 4×4 mechanics-only sheet
  (`mechanics_sheet.yaml`) the area match is at κ ≈ 4, not 1.74. There the CPM mean
  area drops to 18.9 (vs 23.0 on the Sego sheet), with a larger fraction of cells on the
  tissue border. Interface energy per bond (J) vs per length
  (Λ) is commensurable only up to this geometry-dependent factor. (A straight
  order-2 lattice interface crosses ~3 bonds per unit length, so κ ~ 2–4 is the expected
  range.)
* **Neighbour exchange.** The vertex sheet now rearranges (T1s), but far less than the
  CPM's lattice contacts turn over (60–74 of 110 pairs by step 400). CPM contacts include
  flickering diagonal-site contacts, so the two counts are not directly comparable.
* **Extreme tension collapses the small sheet.** At κ = 8 the 4×4 sheet collapses (mean
  area 1.8) instead of reaching a contracted equilibrium.
* **Immune contact geometry.** The CPM immune cells touch the epithelium edge-on in the
  same layer (contact = shared boundary). The vertex overlay touches it face-on from
  above (contact = overlap area), as in Sego's 2-layer model. Kill rates read this contact
  surface, so its units differ between substrates.
* **Weak gradients.** At the shallowest gradient (g = 0.001) the vertex macrophage drifts
  ~40% slower than the CPM one. There the CPM drift is partly noise-rectified, which a
  deterministic tanh law does not reproduce.
* Calibrated at target size 25 (η = 100). Other cell sizes need a re-run of the
  calibration script.
