"""SystemicODEProcess: the shared Price-2015 systemic (global) ODE (spec sections 5, 6).

# Vendored from viva-cpm viva_cpm_studies/influenza/price_ode.py @ b952eec2
# (file unchanged through viva-cpm HEAD 94cf737f). Vendored section below is
# verbatim (module docstring dropped); only the Process wrapper at the end is new.

Integrates the 10 systemic species NB,N,T,X,A,B,P,W,G,O with the spatialized
counts (H,I,M,K,E,L,C,V,F,DH) supplied as step inputs. Outputs:
  ode_state       the 10 integrated states (plus nearby surrogates, held constant)
  sig_1           Sigma1 = a_11*T + a_12*D (D = tot - H - I) frozen at step start,
                  the value the macrophage IL-10 Hill secretion scale consumes
  recruit_drivers per-type inflow/outflow rates from the ODE-driven recruitment
                  laws (see immune.py), the coupling the recruitment component reads
"""
from __future__ import annotations

import re

import numpy as np
from scipy.integrate import solve_ivp

ODE_EPITHELIAL_POPULATION = 250000  # tot_ec_ODE (ImmuneModelInputs.py:27)

# The 10 states actually advanced by the integrator, in vector order.
INTEGRATED_STATES = ("NB", "N", "T", "X", "A", "B", "P", "W", "G", "O")
# Nearby well-mixed surrogates carried in the state dict but held constant here
# (their d/dt = 0 in this process; they are placeholder constants, wired by the
# compile/integration step).
NEARBY_SURROGATES = ("M_nb", "K_nb", "E_nb")

# templated homeostatic constants -> homeostatic_pops_ode key (each * s_v)
_TEMPLATED_HOMEOSTATIC = {"b_m": "macro", "b_k": "nk", "b_p": "apc"}


def _resolve_scale_token(token, s_t: float, s_v: float, s_l: float) -> float:
    """Resolve a scale token (product/quotient of s_t,s_v,s_l powers) to a float.

    ``None`` -> 1.0. Handles e.g. ``"s_t"``, ``"s_t/s_v"``, ``"s_v*s_l"``,
    ``"s_t*s_l/s_v"``, ``"s_t/s_v/s_v"`` and the two reciprocals ``"1/s_l"``
    (a_v) and ``"1/s_v"`` (g_pi) flagged by Task 8.0. Left-to-right, honouring
    ``*`` and ``/`` (all real tokens are pure products/quotients).
    """
    if token is None:
        return 1.0
    text = str(token).replace(" ", "")
    lut = {"s_t": s_t, "s_v": s_v, "s_l": s_l, "1": 1.0}
    parts = re.findall(r"[*/]|[^*/]+", text)
    result = lut[parts[0]]
    i = 1
    while i < len(parts):
        op, name = parts[i], parts[i + 1]
        val = lut[name]
        result = result * val if op == "*" else result / val
        i += 2
    return result


def resolve_constants(
    price_ode: dict,
    *,
    num_epithelial: int,
    s_to_mcs: float = 60.0,
    cell_volume: float = 25.0,
) -> dict:
    """Resolve every ``{base, scale}`` rate constant into a concrete float.

    Applies the spatial-string cellularization scalings (discrepancy #10):
    ``s_t = s_to_mcs/86400``, ``s_v = num_epithelial/250000``,
    ``s_l = 1/250000/cell_volume``. Resolves the templated homeostatic
    populations ``b_m,b_k,b_p = homeostatic_pops_ode[...] * s_v``; ``V0`` is left
    to the caller (passed to :func:`initial_state`).

    Also stashes derived helpers the RHS/derived vars need: the hill exponents,
    the raw scale factors (``_s_t/_s_v/_s_l``) and the unit-corrected T-equation
    denominator constants ``g_2_T = g_2/s_l*s_v`` and ``d_2_T = d_2/s_l*s_v``
    (dossier §2.3 note, lines 138-142 — the spatial model's T eqn uses these in
    place of g_2/d_2 because L is now a spatial-field integral).
    """
    s_t = s_to_mcs / 86400.0
    s_v = num_epithelial / ODE_EPITHELIAL_POPULATION
    s_l = 1.0 / ODE_EPITHELIAL_POPULATION / cell_volume

    homeostatic = price_ode.get("homeostatic_pops_ode", {})
    consts: dict[str, float] = {}

    for name, spec in price_ode["constants"].items():
        scale = spec.get("scale")
        if scale == "templated":
            if name in _TEMPLATED_HOMEOSTATIC:
                consts[name] = homeostatic[_TEMPLATED_HOMEOSTATIC[name]] * s_v
            # V0 (and any other templated value) is supplied by the caller.
            continue
        base = spec.get("base")
        if base is None:
            continue
        consts[name] = base * _resolve_scale_token(scale, s_t, s_v, s_l)

    # Neutrophil homeostatic seeds (ICs NB/N); ODE-scale values * s_v (both 0).
    consts["hs_neutro_blood"] = homeostatic.get("neutro_blood", 0.0) * s_v
    consts["hs_neutro"] = homeostatic.get("neutro", 0.0) * s_v

    # Hill exponents (dossier §2.3): h_m,h_x,h_k,h_e,h_o.
    consts.update({k: float(v) for k, v in price_ode.get("hill_exponents", {}).items()})

    # Raw scale factors, for R (a_rf/s_l*s_v) and the integration cadence.
    consts["_s_t"] = s_t
    consts["_s_v"] = s_v
    consts["_s_l"] = s_l

    # T-equation unit-corrected denominator constants (spatial model, §2.3 note).
    if "g_2" in consts:
        consts["g_2_T"] = consts["g_2"] / s_l * s_v
    if "d_2" in consts:
        consts["d_2_T"] = consts["d_2"] / s_l * s_v

    return consts


def initial_state(
    consts: dict,
    *,
    num_epithelial: int,
    v0: float,
    resist0: float = 0.0,
) -> dict:
    """§2.5 initial conditions for the 10 integrated states + nearby surrogates.

    The spatial vars (H, I, M, ...) are inputs, not part of the integrated state
    vector, so they are not returned here. IC expressions are evaluated in
    dependency order (O -> W -> B -> A -> G).
    """
    tot = float(num_epithelial)
    R = resist0
    b_p = consts["b_p"]
    b_k = consts["b_k"]
    b_m = consts["b_m"]
    h_e = consts["h_e"]
    h_o = consts["h_o"]

    O = consts["b_op"] / consts["mu_o"] * b_p ** h_o / (b_p ** h_o + consts["a_op"] ** h_o)
    W = b_p / consts["mu_w"] * O * consts["b_wo"] / (O + consts["a_wo"])
    B = (consts["b_b"] + consts["b_bp"] * b_p * W * consts["b_0"]) / (
        consts["b_bp"] * b_p * W + consts["mu_b"]
    )
    A = (consts["b_a"] + consts["b_ab"] * B) / consts["mu_a"]
    G = (
        consts["b_go"] * W / (consts["a_go"] + W) * O
        + consts["b_gk"] * W / (consts["a_gk"] + W) * b_k
    ) / consts["mu_g"]

    state = {
        "NB": consts.get("hs_neutro_blood", 0.0),
        "N": consts.get("hs_neutro", 0.0),
        "T": 0.0,
        "X": 0.0,
        "A": A,
        "B": B,
        "P": b_p,
        "W": W,
        "G": G,
        "O": O,
        # nearby surrogates (init_fresh_immune_model): default 0
        "M_nb": 0.0,
        "K_nb": 0.0,
        "E_nb": 0.0,
    }
    return state


def hill(v: float, a: float, h: float) -> float:
    """Hill function (nCoVUtils.hill_equation): 0 at v<=0, else 1/(1+(a/v)^h)."""
    if v <= 0:
        return 0.0
    return 1.0 / (1.0 + (a / v) ** h)


def derived_inputs(state: dict, inputs: dict, consts: dict, num_epithelial: int) -> dict:
    """Compute the §2.2 algebraic assignment rules, frozen at step start.

    ``D = tot - H - I``; ``Sigma1 = a_11*T + a_12*D``;
    ``Sigma2 = Sigma1 + a_21*V/(a_22+V)``; ``DI = tot - H - I - DH``.
    ``R`` is taken from ``inputs`` when the caller supplies it (spatial
    resistance); otherwise computed from F: ``R = F/(a_rf/s_l*s_v + F)``
    (spatial-model form; ``a_rf/s_l*s_v = a_rf_base * s_v``).
    """
    tot = float(num_epithelial)
    H = inputs["H"]
    I = inputs["I"]
    V = inputs.get("V", 0.0)
    DH = inputs.get("DH", 0.0)
    F = inputs.get("F", 0.0)
    T = state["T"]

    D = tot - H - I
    Sigma1 = consts["a_11"] * T + consts["a_12"] * D
    Sigma2 = Sigma1 + consts["a_21"] * V / (consts["a_22"] + V)
    DI = tot - H - I - DH

    R = inputs.get("R")
    if R is None:
        a_rf = consts["a_rf"] / consts["_s_l"] * consts["_s_v"]
        R = F / (a_rf + F) if F > 0 else 0.0

    return {"D": D, "Sigma1": Sigma1, "Sigma2": Sigma2, "DI": DI, "R": R}


def rhs(state_vec: np.ndarray, consts: dict, inputs: dict) -> np.ndarray:
    """d/dt for the 10 integrated states, verbatim from dossier §2.3.

    Only the ``[integrated]`` lines (NB, N, T, X, A, B, P, W, G, O). ``inputs``
    holds the fixed spatialized species and the frozen algebraic vars
    (D, Sigma1, Sigma2, DI, R) from :func:`derived_inputs`.
    """
    c = consts
    NB, N, T, X, A, B, P, W, G, O = state_vec

    M = inputs["M"]
    L = inputs.get("L", 0.0)
    C = inputs.get("C", 0.0)
    V = inputs.get("V", 0.0)
    H = inputs["H"]
    I = inputs["I"]
    K = inputs.get("K", 0.0)
    Sigma2 = inputs["Sigma2"]
    DI = inputs["DI"]
    h_o = c["h_o"]

    # dNB/dt = b_nt*T/(a_nt + a_nl*L + T) - NB*C*g_nc/(C + a_nc) - mu_n*NB
    flux_nb_n = NB * C * c["g_nc"] / (C + c["a_nc"])
    dNB = c["b_nt"] * T / (c["a_nt"] + c["a_nl"] * L + T) - flux_nb_n - c["mu_n"] * NB
    # dN/dt = NB*C*g_nc/(C + a_nc) - mu_n*N
    dN = flux_nb_n - c["mu_n"] * N
    # dT/dt = b_t*M*Sigma2/(Sigma2 + (Sigma2 + (g_1*L+g_2_T)/(L+d_2_T))*(k_1*L+k_2)/(L+d_1)) - mu_t*T
    inner = (c["g_1"] * L + c["g_2_T"]) / (L + c["d_2_T"])
    denom_T = Sigma2 + (Sigma2 + inner) * (c["k_1"] * L + c["k_2"]) / (L + c["d_1"])
    dT = c["b_t"] * M * Sigma2 / denom_T - c["mu_t"] * T
    # dX/dt = b_xn*N/(N + a_xn) - g_xi*I*X - g_xh*H*X - mu_x*X
    dX = c["b_xn"] * N / (N + c["a_xn"]) - c["g_xi"] * I * X - c["g_xh"] * H * X - c["mu_x"] * X
    # dA/dt = b_a + b_ab*B - g_av*A*V - mu_a*A
    dA = c["b_a"] + c["b_ab"] * B - c["g_av"] * A * V - c["mu_a"] * A
    # dB/dt = b_b + b_bp*W*P*(b_0-B) - mu_b*B
    dB = c["b_b"] + c["b_bp"] * W * P * (c["b_0"] - B) - c["mu_b"] * B
    # dP/dt = p_0*(g_pv*V/(a_pv+V) + g_pi*DI)*(g_p + b_pg*G/(a_pg+G)) - mu_p*(P - b_p)
    dP = c["p_0"] * (c["g_pv"] * V / (c["a_pv"] + V) + c["g_pi"] * DI) * (
        c["g_p"] + c["b_pg"] * G / (c["a_pg"] + G)
    ) - c["mu_p"] * (P - c["b_p"])
    # dW/dt = b_wo*O/(a_wo+O)*P - mu_w*W
    dW = c["b_wo"] * O / (c["a_wo"] + O) * P - c["mu_w"] * W
    # dG/dt = b_go*W/(a_go+W)*O + b_gk*W/(a_gk+W)*K - mu_g*G
    dG = (
        c["b_go"] * W / (c["a_go"] + W) * O
        + c["b_gk"] * W / (c["a_gk"] + W) * K
        - c["mu_g"] * G
    )
    # dO/dt = b_op*P^h_o/(P^h_o+a_op^h_o) - mu_o*O
    dO = c["b_op"] * P ** h_o / (P ** h_o + c["a_op"] ** h_o) - c["mu_o"] * O

    return np.array([dNB, dN, dT, dX, dA, dB, dP, dW, dG, dO])


class GlobalODE:
    """Hybrid Price-2015 integrator: advances only the 10 systemic species.

    Reproduces the CC3D driver (``rr.timestep()`` once per MCS): one call to
    :meth:`step` advances ``dt_seconds / seconds_per_mcs`` MCS of the scaled
    (per-MCS) ODE — i.e. one MCS for the default 60 s step. Inputs and the
    derived algebraic vars are held fixed over the step; scipy ``solve_ivp``
    with ``LSODA`` approximates RoadRunner's stiff CVODE integrator.
    """

    def __init__(self, consts: dict, *, num_epithelial: int):
        self.consts = consts
        self.num_epithelial = num_epithelial
        # rate constants carry s_t = s_to_mcs/86400, so they are per-MCS; one
        # MCS spans seconds_per_mcs = s_to_mcs seconds of biological time.
        self.seconds_per_mcs = consts["_s_t"] * 86400.0

    def step(self, state: dict, inputs: dict, dt_seconds: float = 60.0) -> dict:
        derived = derived_inputs(state, inputs, self.consts, self.num_epithelial)
        merged = dict(inputs)
        merged.update(derived)

        y0 = np.array([state[k] for k in INTEGRATED_STATES], dtype=float)
        span = dt_seconds / self.seconds_per_mcs  # in per-MCS model time units

        def fun(t, y):
            return rhs(y, self.consts, merged)

        sol = solve_ivp(
            fun, (0.0, span), y0, method="LSODA", rtol=1e-6, atol=1e-9
        )
        if not sol.success:
            raise RuntimeError(
                f"GlobalODE.step: solve_ivp failed (status={sol.status}): {sol.message}"
            )

        y_final = sol.y[:, -1]
        out = dict(state)
        for i, k in enumerate(INTEGRATED_STATES):
            v = float(y_final[i])
            out[k] = 0.0 if v < 0.0 else v  # clamp tiny numerical negatives
        # nearby surrogates are held constant (d/dt = 0 here) -> carried through
        return out


# --------------------------------------------------------------------------
# Process wrapper (new; not vendored)
# --------------------------------------------------------------------------
from process_bigraph import Process  # noqa: E402

from viva_multicell_blueprint.mechanism.immune import recruit_drivers as _recruit_drivers  # noqa: E402

_INPUT_KEYS = ("H", "I", "M", "K", "E", "DH", "V", "L", "C", "F")


class SystemicODEProcess(Process):
    """Config: ``consts`` (the price_ode params block ``{constants:{NAME:{base,scale}},
    homeostatic_pops_ode, hill_exponents}`` OR an already-resolved flat float map),
    ``num_epithelial``, ``dt_seconds``, optional ``v0`` (initial virus), ``s_to_mcs``,
    ``cell_volume``. The ODE state persists across updates."""

    config_schema = {
        "consts": {"_type": "node", "_default": {}},
        "num_epithelial": {"_type": "integer", "_default": 250000},
        "dt_seconds": {"_type": "float", "_default": 60.0},
        "v0": {"_type": "float", "_default": 0.0},
        "s_to_mcs": {"_type": "float", "_default": 60.0},
        "cell_volume": {"_type": "float", "_default": 25.0},
    }

    def initialize(self, config):
        consts = config.get("consts") or {}
        n = int(config["num_epithelial"])
        if "constants" in consts:
            consts = resolve_constants(
                consts, num_epithelial=n, s_to_mcs=float(config["s_to_mcs"]),
                cell_volume=float(config["cell_volume"]))
        else:
            consts = {k: float(v) for k, v in consts.items()}
        self._consts = consts
        self._n = n
        self._dt = float(config["dt_seconds"])
        self._ode = GlobalODE(consts, num_epithelial=n)
        self._state = initial_state(consts, num_epithelial=n, v0=float(config["v0"]))

    def inputs(self):
        return {k: "float" for k in _INPUT_KEYS} | {"G_ki": "float", "B_ei": "float"}

    def outputs(self):
        # overwrite: the ODE state + recruitment drivers are re-emitted whole each
        # step, atomically replacing the shared store (a bare map[..] apply would
        # drop every key on the empty t=0 store). Readers wire plain map[float].
        return {"ode_state": "overwrite[map[float]]",
                "recruit_drivers": "overwrite[map[float]]", "sig_1": "float"}

    def update(self, inputs, interval=1):
        inp = {k: float(inputs.get(k) or 0.0) for k in _INPUT_KEYS}
        sig_1 = derived_inputs(self._state, inp, self._consts, self._n)["Sigma1"]
        self._state = self._ode.step(self._state, inp, self._dt * float(interval))
        drivers = _recruit_drivers(
            self._state, self._consts, C=inp["C"],
            G_ki=float(inputs.get("G_ki") or 0.0), B_ei=float(inputs.get("B_ei") or 0.0))
        return {"ode_state": dict(self._state), "recruit_drivers": drivers, "sig_1": sig_1}
