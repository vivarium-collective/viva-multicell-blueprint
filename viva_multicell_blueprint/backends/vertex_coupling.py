"""Shared-mechanism -> tyssue coupling for the vertex backend, as BEHAVIORS.

The vertex backend is one self-contained, swappable ``MechanicsBackend`` node, so
its coupling to the shared field / rules / recruitment stores lives *inside* it.
But it is expressed the way vivarium-tyssue's own regulators (``Gillespie``,
``StochasticLineTension``, ``DifferentialAdhesion``) drive the ``EulerSolver``:
each coupler below reads the live tissue + its inputs and returns a list of
**behavior nodes** (``{"func": <BEHAVIOR_MAP name>, ...kwargs}``), which the
backend hands to the solver's ``behaviors`` port. The solver's EventManager runs
them on the real epithelium before integrating, so no coupler ever edits the mesh
directly.

Couplers (inputs -> behavior nodes):

* :func:`field_overlay_behaviors` -- every shared field is OVERLAID on the tissue:
  sampled at each face centroid and written to a ``field_<name>`` face column
  (``apply_gradient``). Fields are never meshed as polygons.
* :func:`migration_behaviors` -- field gradient (chemotaxis) plus a persistent
  random walk (:class:`RandomMotility`) -> per-face ActiveMigration director +
  strength (``apply_gradient``).
* :func:`fate_behaviors` -- fates / latent-pool activations -> ``differentiation``.
* :func:`death_behaviors` / :func:`division_behaviors` -- lineage events.
* :func:`junction_tension_behavior` -- re-tension every junction from the types of
  the two cells it separates (plus an Ornstein-Uhlenbeck fluctuation), via the
  :func:`junction_tension` behavior registered below.
* :func:`confinement_behavior` -- the shared field domain as a hard wall
  (:func:`confine`).

Behavior functions added to the solver's BEHAVIOR_MAP (see
:data:`EXTRA_BEHAVIORS`):

* :func:`junction_tension` -- multi-type differential adhesion. Like
  vivarium-tyssue's ``differential_adhesion`` but with one tension per unordered
  type PAIR (the lowered blueprint adhesion table), recomputed from the CURRENT
  topology every step so a junction created by a T1 picks up the right tension.
"""
from __future__ import annotations

import math

import numpy as np

MEDIUM = "medium"
FIELD_PREFIX = "field_"
MIGRATION_COLS = ("mx", "my", "mz", "migration_strength")


def pair_key(a, b) -> str:
    """Unordered type-pair key, matching the compiler's ``tension_by_pair`` keys."""
    a, b = sorted((str(a), str(b)))
    return f"{a}|{b}"


# --------------------------------------------------------------------------- #
# Behavior functions (run by the solver's EventManager on the live sheet)
# --------------------------------------------------------------------------- #
def _opposite_positions(edge_df) -> np.ndarray:
    from vivarium_tyssue.behaviors.behaviors import opposite_half_edges

    class _S:  # opposite_half_edges only reads sheet.edge_df
        pass

    s = _S()
    s.edge_df = edge_df
    return opposite_half_edges(s)


def junction_tension(sheet, manager, tensions=None, default_tension=0.0,
                     boundary_factor=2.0, sigma=0.0, tau=1.0, dt=1.0, seed=0,
                     type_column="cell_type"):
    """Set every half-edge's ``line_tension`` = adhesion share + fluctuation share.

    * adhesion share (``adhesion_tension`` column): ``tensions[pair_key(a, b)]`` for
      the types of the two faces the junction separates. A half-edge with no
      opposite (the free border of the sheet, or of an overlay island) separates
      its cell from MEDIUM and gets ``boundary_factor * tensions[(type, medium)]``:
      tyssue's LineTension charges ``0.5 * line_tension * length`` per HALF-edge,
      so an interior junction (two half-edges) carries the full tension and a
      border junction (one half-edge) needs the factor 2 to carry it too.
    * fluctuation share (``stochastic_tension`` column): an Ornstein-Uhlenbeck
      process per PHYSICAL junction (both half-edges get the same draw), the
      vertex-model counterpart of the CPM's Metropolis temperature::

          x(t+dt) = x(t) exp(-dt/tau) + sigma sqrt(1 - exp(-2 dt/tau)) N(0, 1)
    """
    edge_df = sheet.edge_df
    if len(edge_df) == 0 or type_column not in sheet.face_df.columns:
        return
    tensions = tensions or {}
    own = sheet.face_df[type_column].reindex(edge_df["face"].to_numpy()).astype(str).to_numpy()
    opp = _opposite_positions(edge_df)
    other = np.where(opp >= 0, own[np.where(opp >= 0, opp, 0)], MEDIUM)
    adhesion = np.array([
        float(tensions.get(pair_key(a, b), default_tension)) for a, b in zip(own, other)
    ])
    adhesion = np.where(opp < 0, boundary_factor * adhesion, adhesion)

    if "stochastic_tension" not in edge_df.columns:
        edge_df["stochastic_tension"] = 0.0
    x = edge_df["stochastic_tension"].to_numpy(dtype=float)
    if sigma > 0.0:
        # one draw per physical junction, keyed by its unordered vertex pair
        srce = edge_df["srce"].to_numpy()
        trgt = edge_df["trgt"].to_numpy()
        lo, hi = np.minimum(srce, trgt), np.maximum(srce, trgt)
        _, junction = np.unique(np.c_[lo, hi], axis=0, return_inverse=True)
        noise = np.random.default_rng(int(seed)).standard_normal(int(junction.max()) + 1)
        decay = math.exp(-float(dt) / float(tau))
        scale = float(sigma) * math.sqrt(1.0 - math.exp(-2.0 * float(dt) / float(tau)))
        x = decay * x + scale * noise[np.asarray(junction).ravel()]
    edge_df["stochastic_tension"] = x
    edge_df["adhesion_tension"] = adhesion
    # tension can fluctuate but never invert into a pulling (negative) junction
    edge_df["line_tension"] = np.maximum(adhesion + x, 0.0)


def overlay_fields(sheet, manager, values=None):
    """Write overlaid field samples ``{column: {face unique_id: value}}`` onto the
    face table (creating the column at 0 for faces not sampled)."""
    fd = sheet.face_df
    for col, by_uid in (values or {}).items():
        if col not in fd.columns:
            fd[col] = 0.0
        mapped = fd["unique_id"].map(by_uid)
        mask = mapped.notna().to_numpy()
        fd.loc[mask, col] = mapped[mask].astype(float).to_numpy()


def confine(sheet, manager, lo=None, hi=None):
    """Hard domain walls: project every vertex back inside ``[lo, hi]`` (the shared
    field domain, as the CPM lattice edge confines its cells)."""
    if lo is None or hi is None:
        return
    vd = sheet.vert_df
    for i, c in enumerate(("x", "y")):
        vd[c] = vd[c].clip(lower=float(lo[i]), upper=float(hi[i]))


EXTRA_BEHAVIORS = {
    "junction_tension": junction_tension,
    "overlay_fields": overlay_fields,
    "confine": confine,
}


# --------------------------------------------------------------------------- #
# Couplers (tissue + port inputs -> behavior nodes)
# --------------------------------------------------------------------------- #
def _sample(grid: np.ndarray, x: float, y: float) -> float:
    ix = int(min(max(round(x), 0), grid.shape[0] - 1))
    iy = int(min(max(round(y), 0), grid.shape[1] - 1))
    return float(grid[ix, iy])


def field_overlay_behaviors(face_df, field_grid) -> list:
    """Overlay each shared field on the tissue: ``field_<name>`` per face."""
    values = {}
    if not field_grid:
        return []
    uids = face_df["unique_id"].to_numpy()
    xs = face_df["x"].to_numpy(dtype=float)
    ys = face_df["y"].to_numpy(dtype=float)
    for name, grid in field_grid.items():
        arr = np.asarray(grid, dtype=float)
        if arr.ndim < 2:
            continue
        values[f"{FIELD_PREFIX}{name}"] = {
            int(u): _sample(arr, x, y) for u, x, y in zip(uids, xs, ys)
        }
    return [{"func": "overlay_fields", "values": values}] if values else []


class RandomMotility:
    """Persistent random walk of selected faces (an active-Brownian director per
    cell): the vertex counterpart of a CPM cell wandering under Metropolis
    fluctuations. Each face carries a polarity angle that diffuses with
    rotational diffusion ``1 / persistence``; the force magnitude is per cell.
    State is keyed by face unique_id, so it survives topology changes."""

    def __init__(self, persistence: float = 10.0, seed: int = 0):
        self.persistence = max(float(persistence), 1e-9)
        self.rng = np.random.default_rng(int(seed))
        self.theta: dict = {}

    def step(self, force_of: dict, dt: float) -> dict:
        """Advance the polarity of every ``{uid: force}`` cell with force > 0;
        return ``{uid: (fx, fy)}``."""
        out = {}
        kick = math.sqrt(2.0 * float(dt) / self.persistence)
        for uid, force in force_of.items():
            if force <= 0.0:
                continue
            th = self.theta.get(uid)
            th = self.rng.uniform(0.0, 2.0 * math.pi) if th is None else th + kick * self.rng.standard_normal()
            self.theta[uid] = th
            out[uid] = (force * math.cos(th), force * math.sin(th))
        return out


def migration_behaviors(face_df, field_grid, chemotaxis, migration_scale,
                        live_uids, random_force=None, saturation=0.0) -> list:
    """ActiveMigration director + strength for every live face, from two sources:

    * chemotaxis -- ``chemotaxis`` is the compiler's ``[{field, type?, strength}]``
      list; a face only feels the entries for its current ``cell_type`` (or untyped
      entries). Its force is ``migration_scale * sum(strength * grad(field))``,
      the gradient sampled at the face centroid. With ``saturation`` > 0 the
      force saturates as ``saturation * tanh(|F| / saturation)`` -- CPM chemotactic
      drift saturates too (a Metropolis flip is accepted with probability <= 1).
    * random motility -- ``random_force`` ``{uid: (fx, fy)}`` from
      :class:`RandomMotility`, added as a force vector.

    ``migration_strength`` is the magnitude of the summed force (ActiveMigration
    normalizes the director). Faces with neither source get strength 0.
    """
    random_force = random_force or {}
    grads = {}
    for name, grid in (field_grid or {}).items():
        arr = np.asarray(grid, dtype=float)
        if arr.ndim >= 2 and chemotaxis:
            gx, gy = np.gradient(arr)[:2]
            grads[name] = (gx, gy)
    if not grads and not random_force:
        return []
    upd = {c: {} for c in MIGRATION_COLS}
    live = set(int(u) for u in live_uids)
    for uid, ctype, x, y in zip(face_df["unique_id"].to_numpy(),
                                face_df["cell_type"].astype(str).to_numpy(),
                                face_df["x"].to_numpy(dtype=float),
                                face_df["y"].to_numpy(dtype=float)):
        uid = int(uid)
        if uid not in live:
            continue
        vx = vy = 0.0
        for ch in (chemotaxis if grads else []):
            want = ch.get("type")
            if want is not None and want != ctype:
                continue
            g = grads.get(ch.get("field"))
            if g is None:
                continue
            s = float(ch.get("strength", 1.0))
            vx += s * _sample(g[0], x, y)
            vy += s * _sample(g[1], x, y)
        vx *= float(migration_scale)
        vy *= float(migration_scale)
        mag = math.hypot(vx, vy)
        if saturation > 0.0 and mag > 0.0:
            k = float(saturation) * math.tanh(mag / float(saturation)) / mag
            vx, vy = vx * k, vy * k
        rx, ry = random_force.get(uid, (0.0, 0.0))
        vx, vy = vx + rx, vy + ry
        upd["mx"][uid] = vx
        upd["my"][uid] = vy
        upd["mz"][uid] = 0.0
        upd["migration_strength"][uid] = math.hypot(vx, vy)
    if not upd["migration_strength"]:
        return []
    return [{
        "func": "apply_gradient",
        "parameter_updates": {
            col: {"dataframe": "face", "update": vals} for col, vals in upd.items()
        },
    }]


def fate_behaviors(fates: dict, uid_of: dict) -> list:
    """Rule fates ``{cell_id: new_type}`` -> ``differentiation`` nodes."""
    out = []
    for cid, new_type in (fates or {}).items():
        uid = uid_of.get(str(cid))
        if uid is not None:
            out.append({"func": "differentiation", "cell_uid": uid, "new_type": str(new_type)})
    return out


def death_behaviors(uids_dead) -> list:
    """Dead cells stop migrating (``is_alive = 0`` gates ActiveMigration)."""
    upd = {int(u): 0 for u in uids_dead}
    if not upd:
        return []
    return [{
        "func": "apply_gradient",
        "parameter_updates": {"is_alive": {"dataframe": "face", "update": upd}},
    }]


def division_behaviors(uids, geom: str) -> list:
    return [{"func": "split_cells", "geom": geom, "cell_uids": [int(u)]} for u in uids]


def confinement_behavior(bounds) -> list:
    if not bounds:
        return []
    return [{"func": "confine", "lo": list(bounds[0]), "hi": list(bounds[1])}]


def junction_tension_behavior(tensions, default_tension, sigma, tau, dt, seed) -> list:
    return [{
        "func": "junction_tension",
        "tensions": dict(tensions),
        "default_tension": float(default_tension),
        "sigma": float(sigma),
        "tau": float(tau),
        "dt": float(dt),
        "seed": int(seed),
    }]
