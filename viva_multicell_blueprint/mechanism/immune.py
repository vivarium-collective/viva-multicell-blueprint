"""Immune component: recruitment coupling + IL-10 Hill secretion scale (spec sections 5, 6).

Immune cells (M/NK/CD8) are SUBSTRATE cells; their movement is mechanics and their
killing is a generic RulesProcess ``kill`` rule. This module holds only the two
shared pieces RulesProcess does not:

  * ODE-driven RECRUITMENT: pure inflow/outflow rate laws (vendored) +
    ``ImmuneRecruitmentProcess``, which reads the ODE ``recruit_drivers`` and emits
    ``spawns`` events ``{"cell_id", "type", "parent"}`` for the backend to place.
  * the IL-10 Hill self-limiting ``macrophage_secretion_scale`` (vendored) the
    Sego immune secretion references.

# Vendored from viva-cpm viva_cpm_studies/influenza/recruitment.py @ b952eec2
#   (hill, *_inflow, *_outflow, ul_rate_to_prob, poisson_inflow_count)
# Vendored from viva-cpm viva_cpm_studies/influenza/signaling.py @ b952eec2
#   (macrophage_secretion_scale, uninfected_il10_scale)
# (both files unchanged through viva-cpm HEAD 94cf737f). Docstrings abbreviated.

Discrepancy #12 (source-faithful): the three recruitment laws are ASYMMETRIC.
CD8+ is driven by APC ``P`` with NO homeostatic baseline; macrophage/NK are driven
by chemokine ``C`` with a ``mu*b`` baseline. Do not symmetrize.
"""
from __future__ import annotations

import math

import numpy as np
from process_bigraph import Process


# --- vendored: recruitment.py ---
def hill(v: float, a: float, h: float) -> float:
    """Hill function (``nCoVUtils.hill_equation``): ``0`` at ``v<=0``, else
    ``1/(1+(a/v)^h)``. Monotone increasing in ``v`` on ``v>0``, saturating to 1."""
    if v <= 0:
        return 0.0
    return 1.0 / (1.0 + (a / v) ** h)


# --- inflow rates (dossier Sec.4b(ii), :1183-1194) ------------------------------

def macrophage_inflow(C: float, consts: dict) -> float:
    """Macrophage inflow rate: ``b_mc*hill(C,a_mc,h_m) + mu_m*b_m``.

    Chemokine(``C``)-driven Hill term plus a homeostatic baseline
    ``mu_m*b_m`` (replenishment toward the tissue-resident population ``b_m``).
    ``hill(0)=0`` -> at ``C<=0`` the rate is exactly the baseline ``mu_m*b_m``.
    """
    return consts["b_mc"] * hill(C, consts["a_mc"], consts["h_m"]) + consts["mu_m"] * consts["b_m"]


def nk_inflow(C: float, consts: dict) -> float:
    """NK inflow rate: ``b_kc*hill(C,a_kc,h_k) + mu_k*b_k``.

    Chemokine(``C``)-driven Hill term plus homeostatic baseline ``mu_k*b_k``
    (same structural form as macrophage). ``hill(0)=0`` -> baseline only at
    ``C<=0``.
    """
    return consts["b_kc"] * hill(C, consts["a_kc"], consts["h_k"]) + consts["mu_k"] * consts["b_k"]


def cd8_inflow(P: float, consts: dict) -> float:
    """CD8+ inflow rate: ``b_ep*hill(P,a_ep,h_e)`` -- discrepancy #12.

    Driven by APC (``P``), NOT chemokine, and with **NO** homeostatic baseline
    term (no ``hs_cd8`` constant): the rate is exactly ``0`` when ``P<=0``. Do
    NOT add a ``+ mu_e*b_e`` term to symmetrize with macrophage/NK.
    """
    return consts["b_ep"] * hill(P, consts["a_ep"], consts["h_e"])


# --- outflow rates (dossier Sec.4b(ii), :1196-1203) -----------------------------

def macrophage_outflow(consts: dict) -> float:
    """Macrophage outflow (attrition) rate: ``mu_m`` (constant)."""
    return consts["mu_m"]


def nk_outflow(G_ki: float, consts: dict) -> float:
    """NK outflow rate: ``G_ki + mu_k``.

    Baseline attrition ``mu_k`` plus the live resistance-weighted infected-load
    term ``G_ki`` (dossier Sec.4a) -- NK are consumed faster where the infected
    load is higher.
    """
    return G_ki + consts["mu_k"]


def cd8_outflow(B_ei: float, consts: dict) -> float:
    """CD8+ outflow rate: ``B_ei + mu_e``.

    Baseline attrition ``mu_e`` plus the resistance-weighted infected-load term
    ``B_ei`` (the CD8+ analogue of NK's ``G_ki``).
    """
    return B_ei + consts["mu_e"]


# --- rate -> probability / count (source-faithful stochastic conversion) ---------

def ul_rate_to_prob(rate: float) -> float:
    """``ImmuneModelLib.ul_rate_to_prob``: ``1 - exp(-rate)`` (per-MCS unit-less
    rate -> Bernoulli probability). Used for outflow removals."""
    return 1.0 - math.exp(-rate)


def poisson_inflow_count(rate: float, rng) -> int:
    """Number of new cells to seed for a per-MCS inflow ``rate`` -- the source's
    ``inflow_by_type`` draw (``ViralInfectionVTMSteppables`` :1205-1212),
    reproduced verbatim over an explicit seeded ``numpy`` ``Generator`` for
    determinism.

    Each iteration draws a fresh uniform and accepts (increments the count) with
    probability ``1 - exp(-rate)*sum_{j=0}^{k} rate^j/j!`` = ``P(X >= k+1)`` for
    ``X ~ Poisson(rate)`` -- the source's sequential Poisson-tail sampler. For
    the tiny per-MCS rates at small patch scale this is ~always ``0``.
    """
    if rate <= 0.0:
        return 0
    num_add = 0
    exp_term = math.exp(-rate)
    sum_term = 1.0
    while rng.random() < 1.0 - exp_term * sum_term:
        num_add += 1
        sum_term += rate ** num_add / math.factorial(num_add)
    return num_add


# --- vendored: signaling.py ---
def macrophage_secretion_scale(il10_local: float, sig_1: float, g_1: float,
                                g_2: float, d_2: float) -> float:
    """The shared IL-10-Hill secretion-scale factor gating macrophage
    chemokine + IL-10 secretion: `sig_1 / (sig_1 + (g_1*L + g_2) / (L + d_2))`,
    `L` = `il10_local` (clamped to >= 0 so a negative, e.g. numerically-noisy,
    local IL-10 reading never flips the ratio term negative).

    Monotone DECREASING in `il10_local` for `g_1, d_2 > 0`: more local IL-10
    -> a larger `(g_1*L+g_2)/(L+d_2)` ratio term -> a smaller Hill factor
    (self-limiting secretion as IL-10 builds up). Always >= 0 for `sig_1,
    g_1, g_2, d_2 > 0` and `il10_local >= 0` (a ratio of two non-negative
    quantities); returns 0.0 in the degenerate case where the denominator is
    not positive (e.g. `sig_1 <= 0`), rather than raising or returning a
    negative/undefined value.
    """
    L = max(0.0, il10_local)
    denom = sig_1 + (g_1 * L + g_2) / (L + d_2)
    if denom <= 0.0:
        return 0.0
    return max(0.0, sig_1 / denom)


def uninfected_il10_scale(resist: float) -> float:
    """Uninfected (H) cell IL-10 secretion scale: `1 - resist`, clamped to
    [0, 1] (source: `sec_amount = mu_l * b_lh * (1 - resist)`, applied per
    UNINFECTED cell -- see `params.yaml`'s `il10.mu_l`/`il10.b_lh`).
    `resistance.cell_resistance` already returns values in [0, 1), but this
    function clamps independently so it stays well-defined for any caller."""
    return min(1.0, max(0.0, 1.0 - resist))


# --------------------------------------------------------------------------
# New (not vendored): driver bundle + recruitment Process
# --------------------------------------------------------------------------
def recruit_drivers(state: dict, consts: dict, *, C: float, G_ki: float = 0.0,
                    B_ei: float = 0.0) -> dict:
    """Per-type inflow/outflow rates from the ODE state (P) and chemokine (C)."""
    P = float(state["P"])
    return {
        "macrophage_inflow": macrophage_inflow(C, consts),
        "nk_inflow": nk_inflow(C, consts),
        "cd8_inflow": cd8_inflow(P, consts),
        "macrophage_outflow": macrophage_outflow(consts),
        "nk_outflow": nk_outflow(G_ki, consts),
        "cd8_outflow": cd8_outflow(B_ei, consts),
        "P": P,
        "C": float(C),
    }


class ImmuneRecruitmentProcess(Process):
    """Turns ``recruit_drivers`` inflow rates into spawn events.

    Config: ``types`` {driver key -> cell type} (default M/NK/CD8), ``seed``,
    ``id_prefix``, ``start_index`` (first new id number), ``max_per_step`` cap per type.
    Counts use the source's sequential Poisson-tail sampler (``poisson_inflow_count``).
    Placement is left to the backend (events carry no position)."""

    config_schema = {
        "types": {"_type": "map[string]", "_default": {
            "macrophage_inflow": "M", "nk_inflow": "NK", "cd8_inflow": "CD8"}},
        "seed": {"_type": "integer", "_default": 0},
        "id_prefix": {"_type": "string", "_default": "rec"},
        "start_index": {"_type": "integer", "_default": 0},
        "max_per_step": {"_type": "integer", "_default": 1000},
    }

    def initialize(self, config):
        self._rng = np.random.default_rng(int(config.get("seed", 0)))
        self._types = dict(config.get("types") or {})
        self._prefix = config.get("id_prefix", "rec")
        self._next = int(config.get("start_index", 0))
        self._cap = int(config.get("max_per_step", 1000))

    def inputs(self):
        return {"recruit_drivers": "map[float]"}

    def outputs(self):
        # ``spawns``: per-event list (kept for provenance / lineage).
        # ``activations``: active_type -> count to ACTIVATE this step. The MechanicsBackend
        # ``activations`` input consumes this to activate that many pre-placed LATENT cells
        # (Task 20), closing the T13-GAP where ``spawns`` was emitted but never consumed.
        # ``overwrite`` so the full count map atomically REPLACES the shared store each
        # step (a bare map apply would merge-and-leave stale counts); the backend reads a
        # plain ``map[integer]`` input, matching the mechanics overwrite<->plain pattern.
        return {"spawns": "list[node]", "activations": "overwrite[map[integer]]"}

    def update(self, inputs, interval=1):
        drivers = inputs.get("recruit_drivers") or {}
        spawns = []
        activations: dict[str, int] = {}
        for key in sorted(self._types):
            n = min(poisson_inflow_count(float(drivers.get(key, 0.0)), self._rng), self._cap)
            active_type = self._types[key]
            activations[active_type] = activations.get(active_type, 0) + n
            for _ in range(n):
                spawns.append({"cell_id": f"{self._prefix}{self._next}",
                               "type": active_type, "parent": None})
                self._next += 1
        return {"spawns": spawns, "activations": activations}
