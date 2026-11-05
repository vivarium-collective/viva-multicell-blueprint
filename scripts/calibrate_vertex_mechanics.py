#!/usr/bin/env python3
"""Calibrate the vertex backend's free mechanics knobs against the CPM.

The two substrates share no units (Metropolis temperature vs viscous vertex
dynamics), so the vertex model's fluctuation/mobility knobs are fitted to match
measurable CPM behaviour on the SAME blueprint initial state, mechanics only (no
field / rule / ODE coupling -- each backend is driven directly):

* ``sheet_motility``  -> epithelial MSD relative to the sheet centre of mass,
  CAPPED at the largest value whose mesh never folds (inverted / self-intersecting
  faces) over a long run -- validity is a hard constraint, the MSD gap is reported
* ``motility``        -> latent-immune random-walk displacement (median |d|),
  CAPPED the same way (the immune cells are polygons IN the sheet, so their walk
  deforms the mesh too)
* ``tension_scale``   -> epithelial mean area (junction tension Lambda_ab =
  tension_scale * J_ab contracts cells against their area elasticity)
* ``tension_noise``   -> epithelial area coefficient of variation
* ``migration_scale`` + ``chemotaxis_saturation`` -> the chemotactic drift-speed
  vs gradient curve of one active macrophage in a fixed linear virus field

Everything else is NOT fitted: area elasticity is energy-matched (K = 2
lambda_volume), the per-pair tension RATIOS are the blueprint's J ratios (only the
overall scale is fitted), viscosity is set for Euler stability. Results are averaged
over seeds and written to the adhesion-calibration study's results/.

Usage: python scripts/calibrate_vertex_mechanics.py [--fast]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile.to_cpm import cpm_mechanics_node
from viva_multicell_blueprint.compile.to_vertex import vertex_mechanics_node
from viva_multicell_blueprint.core import build_core

ROOT = Path(__file__).resolve().parents[1]
SEGO = ROOT / "workspace" / "blueprints" / "influenza_sego2022.yaml"
OUT = ROOT / "workspace" / "studies" / "adhesion-calibration" / "results" / "vertex_mechanics_calibration.json"
IMMUNE = {"M", "NK", "CD8", "M_latent", "NK_latent", "CD8_latent"}


def _backend(kind, blueprint, overrides=None, seed=0):
    from viva_multicell_blueprint.backends.cpm_backend import CPMBackend
    from viva_multicell_blueprint.backends.vertex_backend import VertexBackend

    if kind == "cpm":
        node = cpm_mechanics_node(blueprint)
        node["config"]["backend_params"]["potts"] = {"seed": seed}
        return CPMBackend(node["config"], core=build_core())
    overrides = dict(overrides or {})
    if "tension_scale" in overrides:     # a lowering knob: re-lower the adhesion
        d = copy.deepcopy(blueprint.to_dict())
        d.setdefault("mechanics_mapping", {})["tension_scale"] = overrides.pop("tension_scale")
        blueprint = Blueprint.from_dict(d)
    node = vertex_mechanics_node(blueprint)
    node["config"]["backend_params"].update(overrides)
    node["config"]["backend_params"]["seed"] = seed
    return VertexBackend(node["config"], core=build_core())


def mechanics_observables(kind, blueprint, n_steps, overrides=None, seed=0) -> dict:
    """Pure-mechanics run (no inputs): epithelial MSD, immune displacement, area."""
    b = _backend(kind, blueprint, overrides, seed)
    out0 = b._emit()
    out = out0
    for _ in range(n_steps):
        out = b.update({}, 1.0)
    epi = [i for i, t in out0["types"].items() if t not in IMMUNE]
    imm = [i for i, t in out0["types"].items() if t in IMMUNE]

    def rel(o):
        p = np.array([o["positions"][i] for i in epi], dtype=float)
        return p - p.mean(0)

    msd = float(np.mean(np.sum((rel(out) - rel(out0)) ** 2, 1)))
    d_imm = [float(np.hypot(*np.subtract(out["positions"][i], out0["positions"][i]))) for i in imm]
    areas = np.array([out["sizes"][i] for i in epi], dtype=float)
    return {"epi_msd": msd, "immune_disp_median": float(np.median(d_imm)) if d_imm else 0.0,
            "area_mean": float(areas.mean()), "area_cv": float(areas.std() / areas.mean())}


def chemotaxis_drift(kind, blueprint, gradient, n_steps, overrides=None, seed=0) -> float:
    """x-velocity (um/step) of ONE active macrophage in a fixed linear virus field."""
    d = copy.deepcopy(blueprint.to_dict())
    d["initial_state"] = {"kind": "blocks", "block": {"type": "M", "origin": [15, 25], "shape": [1, 1],
                                                      "spacing": 5, "size": 25}}
    bp = Blueprint.from_dict(d)
    b = _backend(kind, bp, overrides, seed)
    nx, ny = 50, 50
    field = {"virus": (gradient * np.tile(np.arange(nx, dtype=float)[:, None], (1, ny))).tolist()}
    x0 = b._emit()["positions"]["1"][0]
    out = None
    for _ in range(n_steps):
        out = b.update({"field_grid": field}, 1.0)
    return (out["positions"]["1"][0] - x0) / n_steps


def n_folds(blueprint, overrides, n_steps, seeds) -> int:
    """Worst count of inverted / self-intersecting faces (epithelial AND in-sheet
    immune) over a mechanics-only vertex run (sampled every 10 steps), across seeds.
    A folded vertex mesh is invalid geometry, so any fitted motility must keep this
    at 0."""
    from viva_multicell_blueprint.backends.vertex_mesh import n_invalid_faces

    worst = 0
    for seed in seeds:
        b = _backend("vertex", blueprint, overrides, seed)
        for k in range(1, n_steps + 1):
            out = b.update({}, 1.0)
            if k % 10:
                continue
            polys = [f["xy"] for f in out["footprints"].values() if f["xy"]]
            worst = max(worst, n_invalid_faces(polys))
            if worst:
                return worst
    return worst


def _mean(dicts):
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    n = 30 if args.fast else 60
    seeds = (0, 1) if args.fast else (0, 1, 2, 3, 4)
    bp = Blueprint.load(SEGO)

    cpm = _mean([mechanics_observables("cpm", bp, n, seed=s) for s in seeds])
    print("CPM target:", {k: round(v, 3) for k, v in cpm.items()})

    # 1) Fit each knob on its own observable, in sequence: sweep, then interpolate
    #    the monotone response to hit the CPM target. (The immune cells sit in the
    #    sheet, so the knobs interact; each fit uses the knobs fitted before it.)
    def obs(ov):
        return _mean([mechanics_observables("vertex", bp, n, ov, seed=s) for s in seeds])

    def fit_1d(knob, values, metric, base):
        rows = []
        for v in values:
            o = obs({**base, knob: v})
            rows.append({knob: v, **o})
            print(f"  {knob}={v}: {metric}={o[metric]:.3f} (target {cpm[metric]:.3f})")
        xs = np.array([r[knob] for r in rows], float)
        ys = np.array([r[metric] for r in rows], float)
        order = np.argsort(ys)
        val = float(np.interp(cpm[metric], ys[order], xs[order]))
        return val, rows

    sweeps = {}
    base = {"tension_noise": 0.0, "sheet_motility": 0.0, "motility": 0.0,
            "chemotaxis_saturation": 0.0, "tension_scale": 1.0}
    # junction tension Lambda_ab = tension_scale * J_ab sets the relaxed cell area
    # (tension contracts cells against area elasticity) -> fit it on the mean area
    ts_vals = [1.0, 2.0, 3.0, 4.0, 6.0] if not args.fast else [1.0, 4.0]
    base["tension_scale"], sweeps["tension_scale"] = fit_1d("tension_scale", ts_vals, "area_mean", base)
    # HARD CONSTRAINT: the mesh must never fold. Step each motility down from its
    # fit until a long run stays valid; report the observable it then reaches.
    n_valid, valid_seeds = (100, (0,)) if args.fast else (400, (0, 1, 2))

    def fold_free(knob, fit):
        v = fit
        while v > 1.0 and n_folds(bp, {**base, knob: v}, n_valid, valid_seeds):
            print(f"  {knob}={v:.1f}: folds the mesh -> lower")
            v *= 0.8
        print(f"  {knob}: fit {fit:.1f}, fold-free cap {v:.1f}")
        return v

    m_vals = [25, 50, 100, 200, 300] if not args.fast else [50, 200]
    mot_fit, sweeps["motility"] = fit_1d("motility", m_vals, "immune_disp_median", base)
    base["motility"] = fold_free("motility", mot_fit)
    s_vals = [40, 55, 70, 85, 100] if not args.fast else [60, 90]
    msd_fit, sweeps["sheet_motility"] = fit_1d("sheet_motility", s_vals, "epi_msd", base)
    sm = fold_free("sheet_motility", msd_fit)
    base["sheet_motility"] = sm
    validity = {"disp_fit_motility": mot_fit, "fold_free_motility": base["motility"],
                "msd_fit_sheet_motility": msd_fit, "fold_free_sheet_motility": sm,
                "validity_steps": n_valid, "validity_seeds": list(valid_seeds)}
    t_vals = [0.0, 5.0, 10.0, 20.0] if not args.fast else [0.0, 10.0]
    _, sweeps["tension_noise"] = fit_1d("tension_noise", t_vals, "area_cv", base)
    # area CV responds weakly and non-monotonically to the noise (see sweeps): take
    # the closest sweep value, but only if it beats no-noise by more than seed noise
    # (0.01 in CV) -- otherwise the knob is not identifiable and stays off.
    tn_rows = sweeps["tension_noise"]
    closest = min(tn_rows, key=lambda r: abs(r["area_cv"] - cpm["area_cv"]))
    zero = next(r for r in tn_rows if r["tension_noise"] == 0.0)
    gain = abs(zero["area_cv"] - cpm["area_cv"]) - abs(closest["area_cv"] - cpm["area_cv"])
    base["tension_noise"] = closest["tension_noise"] if gain > 0.01 else 0.0
    # the adopted noise must keep the mesh valid too; step it down if it folds
    for tn in sorted({r["tension_noise"] for r in tn_rows if r["tension_noise"] <= base["tension_noise"]},
                     reverse=True):
        base["tension_noise"] = tn
        if not n_folds(bp, base, n_valid, valid_seeds):
            break
    validity["tension_noise"] = base["tension_noise"]
    fitted = {k: round(float(base[k]), 3)
              for k in ("tension_scale", "tension_noise", "sheet_motility", "motility")}
    best = obs(fitted)
    rows = sweeps
    print("fitted:", fitted, "->", {k: round(v, 3) for k, v in best.items()})

    # 2) chemotactic drift -> (migration_scale, chemotaxis_saturation). CPM drift
    #    SATURATES with the gradient (flip acceptance <= 1); the vertex force is
    #    F = F_sat tanh(scale * strength * g / F_sat) and velocity = F / drag, so fit
    #    v(g) = V_sat tanh(c g / V_sat) to the CPM curve, then convert with the
    #    measured drag.
    grads = (0.001, 0.003, 0.01, 0.03)
    cpm_v = [float(np.mean([chemotaxis_drift("cpm", bp, g, n, seed=s) for s in seeds])) for g in grads]
    strength = next(float(c["strength"]) for f in bp.fields for c in (f.get("chemotaxis") or [])
                    if c.get("type") == "M")
    # drag is a property of the viscosity alone: measure it noise-free, with a probe
    # force well above any fluctuation.
    probe = 10.0
    quiet = {**fitted, "motility": 0.0, "tension_noise": 0.0, "chemotaxis_saturation": 0.0}
    v_probe = chemotaxis_drift("vertex", bp, 0.001, n, {**quiet, "migration_scale": probe})
    drag = probe * strength * 0.001 / v_probe            # force per unit velocity
    best_fit = None
    for v_sat in np.linspace(0.2, 0.5, 31):
        for c in np.linspace(5, 200, 196):
            pred = v_sat * np.tanh(c * np.asarray(grads) / v_sat)
            sse = float(np.sum((pred - np.asarray(cpm_v)) ** 2))
            if best_fit is None or sse < best_fit[0]:
                best_fit = (sse, float(v_sat), float(c))
    _, v_sat, c = best_fit
    migration_scale = drag * c / strength
    saturation = drag * v_sat
    ov = {**fitted, "motility": 0.0, "migration_scale": migration_scale,
          "chemotaxis_saturation": saturation}
    v_fit = [chemotaxis_drift("vertex", bp, g, n, ov) for g in grads]
    print("chemotaxis CPM v:", np.round(cpm_v, 4), "vertex v:", np.round(v_fit, 4),
          f"scale={migration_scale:.4g} saturation={saturation:.4g} drag={drag:.4g}")

    result = {
        "n_steps": n, "seeds": list(seeds),
        "cpm_target": cpm, "sweeps": rows, "fitted": {**fitted, "migration_scale": migration_scale,
                   "chemotaxis_saturation": saturation},
        "fitted_observables": best,
        "mesh_validity": validity,
        "chemotaxis": {"gradients": list(grads), "cpm_velocity": cpm_v,
                       "vertex_velocity": v_fit, "drag": drag,
                       "v_sat": v_sat, "slope": c},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
