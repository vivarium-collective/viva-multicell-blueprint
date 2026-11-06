# Mechanics of the three substrates — equations, parameters, and how they are matched

The cross-paradigm investigation runs ONE blueprint on three mechanical substrates
(CPM, vertex, and a center-based model). The shared mechanism (fields, rules, systemic
ODE, recruitment) is byte-identical across all three compiles; **only the mechanics
differ**. This page defines the mechanical models term by term, lists every parameter
(including per-cell-type and per-type-pair variations) as used for the Sego 2022
influenza blueprint
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
| J(a,b) | contact energy per neighbour bond | table §4 | blueprint `adhesion` (`J = strength`, exact) |
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
medium). Immune cells (M/NK/CD8 and their latent pools) are members of the **same
sheet, in the same plane**, as they are in the CPM lattice. Each one is attached to the
tissue on the free lattice site at the sheet margin nearest its blueprint centroid
(`attach_to_margin`), so it is bound to the tissue rather than floating free. It is a
**smaller polygon**: it relaxes to A₀ = `immune_area_scale` × target size (0.5), so it
can squeeze along and between epithelial cells by neighbour exchange. It also carries a
shape stiffness Γ = `immune_perimeter_elasticity` (20) with P₀ = 3.9·√A₀. A latent
cell has no adhesion row (tension 0), so it wets the high-tension tissue border, as
the CPM latent cells do. Unlike a lattice cell, though, a polygon can thin into a
sliver that folds, and the shape stiffness keeps it compact. (Free overlay islands
above the sheet, Sego's 2-layer picture, remain available opt-in via
`mechanics_mapping.vertex_overlay_types`; none by default.) The shared fields are
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

**Topology.** `auto_reconnect` is on: a junction shorter than `threshold_length`
(0.2 × hexagon side = 0.62) merges into a 4-way vertex that detaches along the other
axis (T1 neighbour exchange; T3 at the border). A rank-4 vertex detaches at p₄ = 1 per
unit time and a rank-5+ vertex at tyssue's p₅₊ = 0.01. At tyssue's default p₄ = 0.1, a
merged rosette stood for ~10 steps, so exchanges stalled half-done; the small immune
polygons (side ≈ 2.2) also need the larger threshold to swap before a short junction
inverts. The playback's vertex panel counts the neighbour exchanges frame by frame. Each coupled
step is integrated in 4 solver sub-steps (`substeps`), so per-update vertex motion stays
well below the T1 threshold. The shared field domain is a hard wall (`confine`
behavior), as the lattice edge is for the CPM.

**Mesh validity is a hard constraint.** An inverted or self-intersecting face is broken
geometry, not a cell shape. Every fitted knob must keep the sheet fold-free over 3 seeds
× 400 steps (§5). `tests/test_vertex_mesh_validity.py` pins this on the Sego run.

**Coupling** to the shared mechanism goes entirely through vivarium-tyssue *behaviors*
handed to the solver's `behaviors` port, the way the Gillespie regulator drives the
`EulerSolver`: `overlay_fields` (field → face columns), `apply_gradient` (migration
director/strength; `is_alive` on death), `differentiation` (rule fates and latent →
active immune activation), `split_cells` (division), and `junction_tension` (per-pair
tension recomputed on the current topology every step, so a junction created by a T1
gets the right tension). See `viva_multicell_blueprint/backends/vertex_coupling.py`.

**Contact surface** (what the rules read as `shared_surface_by_type`): in the sheet, the
shared junction length per neighbour type. This includes the in-sheet immune cells, as
in the CPM, where contact is shared boundary. For opt-in overlay cells it is the
**polygon overlap area** with each cell underneath, credited both ways.

## 3. Center-based model (viva-munk / pymunk) — `MunkBackend`

The center-based paradigm (the third substrate) represents each cell as a **point-center
with a radius** — a `pymunk` circular rigid body (Chipmunk2D) rather than CPM voxels or
vertex polygons. One blueprint cell ↦ one circular body at its centroid, radius
`r = sqrt(size / π)` so the emitted area equals the blueprint `size` exactly.

**Dynamics.** Bodies integrate under damped 2D rigid-body physics: pairwise collision
(hard-disc repulsion on overlap, so the sheet relaxes to a non-overlapping packing),
velocity damping (`damping_per_second`, default 0.1 — the regime is overdamped, as
befits a crawling tissue, not ballistic), optional active-Brownian jitter
(`jitter_per_second`, default 0 for a deterministic comparison), and chamber walls sized
to the blueprint domain. There is no volume/surface *energy* as in CPM/vertex; cell area
is fixed by the disc radius and held by the hard-disc constraint.

**Field seam (as for CPM).** The authoritative reaction-diffusion field lives in the
shared `FieldProcess`; viva-munk's own field processes are **not** run. Each step the
shared `field_grid` is sampled at every cell center and its local gradient drives a
**chemotactic velocity** (per-type strength, from the blueprint `fields[].chemotaxis`,
injected directly on the body since `pymunk` thrust only acts on segment cells). The
field is read, never re-diffused here.

**Contact surface** (`shared_surface_by_type`): the **chord length of the overlap lens**
of two discs within the touching tolerance `contact_eps` — the center-based analog of
the CPM shared face / vertex junction length. Neighbours are cells whose discs come
within `contact_eps` of touching. Footprints are emitted as polygon discs so the
rasterizer draws each cell as a filled circle with an outline.

| Symbol | Meaning | Value (Sego blueprint) | Source |
|---|---|---|---|
| r | cell radius | sqrt(size/π) ≈ 2.82 (size 25) | blueprint `cell_types[].target_size` |
| substeps | physics sub-steps per coupled step | 10 | `mechanics_mapping.substeps` |
| damping | velocity damping per second | 0.1 | `mechanics_mapping.damping_per_second` |
| jitter | active-Brownian impulse std | 0 (off) | `mechanics_mapping.jitter_per_second` |
| contact_eps | touching tolerance (um) | 0.75 | `mechanics_mapping.contact_eps` |
| chemotaxis | per-type up-gradient velocity | blueprint `fields[].chemotaxis` strengths | shared field (sampled) |

**Not yet energy-calibrated to the other two.** The vertex knobs are calibrated to
reproduce CPM behaviour (§5); the center-based model is **not** fitted to either — its
hard-disc packing is a genuinely independent third representation. This is deliberate:
the cross-paradigm result (§ below / the equivalence study) shows the three paradigms
**agree on fate** (rule-driven, substrate-independent) while the center-based model is
the **most spatially divergent** of the three. Calibrating a center-based↔CPM area/
motility match is future work, parallel to §5.

## 4. Parameters, per type and per type pair (Sego blueprint)

| Parameter | CPM | Vertex | How the vertex value is set |
|---|---|---|---|
| Target size | V_T = 25 | A_0 = 25 | identical (blueprint `target_size`) |
| Size stiffness | λ_V = 1 | K = 2 | **energy-matched**: CPM λ(V−V_T)² ≡ tyssue K/2(A−A_0)² ⇒ K = 2λ_V, per type |
| Surface stiffness | λ_S = 0 | Γ = 0 | energy-matched (Γ = 2λ_S, P_0 = S_T), per type |
| Adhesion | J(a,b) | Λ_ab = κ·J(a,b) | **lowered** per pair; one global scale κ = `tension_scale` **calibrated** (§5) |
| Missing pair | J = 0 | Λ = 0 | identical (viva-cpm default) |
| Fluctuations | T = 10 | f_mot, σ_Λ | **calibrated** (§5) |
| Chemotaxis | λ_chem | s_mig·λ_chem, F_sat | λ_chem ratios identical; s_mig and F_sat **calibrated** (§5) |
| Mobility | (implicit in MCS) | η = 2·K·A_0 = 100 | set for explicit-Euler stability (the area stiffness ~K·A_0 must stay below η/dt) |
| Neighbour exchange | implicit (lattice) | T1 at ℓ < 0.2·side = 0.62, p₄ = 1 | `threshold_length` scaled to the hexagon side |
| Immune placement | same lattice, at margin | same sheet, attached at the margin site nearest the centroid | `immune_types` = immune types + latent pools |
| Immune size | V_T = 25 | A₀ = 12.5 (Γ = 20) | `immune_area_scale` = 0.5, `immune_perimeter_elasticity` = 20 (representation, not fitted) |

Per-type-pair adhesion (blueprint `strength` = CPM J; vertex Λ = 1.683·J):

| Pair | J | Λ | | Pair | J | Λ |
|---|---|---|---|---|---|---|
| epithelial–epithelial (H/I/D × H/I/D) | 5 | 8.4 | | M–M, NK–NK, CD8–CD8 | 25 | 42.1 |
| epithelial–medium (H/I/D) | 25 | 42.1 | | M–NK, M–CD8 | 10 | 16.8 |
| immune–medium (M/NK/CD8) | 10 | 16.8 | | NK–CD8 | 25 | 42.1 |
| M–H, M–I, M–D | 20 | 33.7 | | NK–I, CD8–I | 10 | 16.8 |
| NK–H, NK–D, CD8–H, CD8–D | 20 | 33.7 | | any latent type × anything | — (0) | 0 |

On the vertex side, immune–epithelium and immune–immune tensions act wherever those
cells share a junction. The immune cells are in the sheet, so these rows are live, as
in the CPM.

Per-type motility group (vertex): epithelial cells (H/I/D) f_mot = 26.6; immune cells
(immune + latent, in the sheet) f_mot = 56.1. Both are fold-free caps (§5). Persistence
τ_p = 10 steps for both.

## 5. Calibration — making the free vertex knobs reproduce CPM behaviour

The temperature T and the vertex mobility/fluctuation knobs share no units, so the
vertex knobs that have no CPM counterpart are **fitted to CPM behaviour** on the same
Sego initial state, mechanics only, averaged over 5 seeds × 60 steps
(`scripts/calibrate_vertex_mechanics.py` →
`workspace/studies/adhesion-calibration/results/vertex_mechanics_calibration.json`;
values in `compile/to_vertex.py: CALIBRATED_DEFAULTS`):

| Knob | Fitted to | CPM | Vertex (fitted) |
|---|---|---|---|
| `tension_scale` κ = 1.683 | mean epithelial area | 23.0 | 23.0 |
| `sheet_motility` = 26.6 (**cap**; MSD fit = 81) | epithelial MSD about the sheet centre, 60 steps | 9.5 µm² | **2.2 µm²** |
| `motility` = 56.1 (**cap**; displacement fit = 214) | median latent-immune displacement, 60 steps (walls on) | 9.4 µm | **4.6 µm** |
| `migration_scale` = 3.54, `chemotaxis_saturation` = 105 | macrophage drift speed vs virus gradient g = 0.001 / 0.003 / 0.01 / 0.03 | 0.098 / 0.145 / 0.331 / 0.350 | 0.058 / 0.163 / 0.327 / 0.350 µm/step |
| `tension_noise` = 5 (largest fold-free step) | epithelial area CV | 0.093 | 0.038 (not reachable; see below) |

**What is *not* equivalent, honestly:**

* **Epithelial wandering is ~4× lower, immune wandering ~2× lower.** Matching the CPM's
  epithelial MSD needs a sheet motility of ~81, but at that force (and at any force above
  ~27 once the in-sheet immune cells walk too) the vertex sheet folds within a few
  hundred steps. The in-sheet immune walk is capped the same way (56 vs a fit of 214):
  a welded polygon cannot wander as freely as the old free overlay island. Folding persisted with smaller time steps, a
  perimeter (shape) stiffness, faster rosette resolution and larger T1 thresholds.
  tyssue's reconnection resolves short junctions, not a vertex sweeping across a
  non-adjacent edge. So the motility is capped at the largest fold-free value, and the
  vertex epithelium wanders less (MSD 2.2 vs 9.5 µm² over 60 steps). It still exchanges
  neighbours: 4–11 of 85 epithelial pairs swap by step 400, gains ≈ losses (genuine T1s).
* **Area dispersion.** The vertex area CV (0.038) stays below the CPM's (0.093).
  Tension noise raises it, but the levels that would get closer (10, 20) fold the mesh,
  so it stays at the largest fold-free step, 5.
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
* **Extreme tension collapses the small sheet.** At κ = 8 the 4×4 sheet collapses
  completely (every junction merges away; area 0) instead of reaching a contracted
  equilibrium. The backend reports every cell vanished rather than failing.
* **Immune geometry.** Both substrates now have the immune cells edge-on in the tissue
  plane, with contact = shared boundary. On the vertex side, though, they are attached to
  the margin at t0 (the CPM leaves a medium gap of up to one cell diameter, so
  `immune_attach_max` ≈ 5.6 µm). They are also half the CPM area, a representation
  choice so they can squeeze along the tissue.
* **Weak gradients.** At the shallowest gradient (g = 0.001) the vertex macrophage drifts
  ~40% slower than the CPM one. There the CPM drift is partly noise-rectified, which a
  deterministic tanh law does not reproduce.
* Calibrated at target size 25 (η = 100). Other cell sizes need a re-run of the
  calibration script.
