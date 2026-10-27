"""Cross-backend comparison metrics (spec s8 scoring rule, s9 capstone).

A *run* is a list of per-step frames (MechanicsBackend port dicts keyed by
blueprint cell id; optional ``field_integrals``: name -> float). A cell's state
is ``"dead"`` if ``alive`` is False, its type if alive, or ``None`` if absent
from the frame (absent cells are treated as dead for scoring; cells that truly
vanished from bookkeeping are already excluded by ``registry.per_cell_scored_ids``).
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

DEAD = "dead"


def cell_state(frame, cid):
    if not frame.get("alive", {}).get(cid, False):
        return DEAD
    return str(frame.get("types", {}).get(cid, DEAD))


def per_cell_agreement(runA, runB, registry) -> dict:
    """Over the t0 cohort (minus vanished), fraction in the same state per step.
    Alive-vs-dead is a DISAGREEMENT (spec s8)."""
    ids = sorted(registry.per_cell_scored_ids())
    n_steps = min(len(runA), len(runB))
    per_step = []
    for t in range(n_steps):
        if not ids:
            per_step.append(float("nan"))
            continue
        same = sum(cell_state(runA[t], i) == cell_state(runB[t], i) for i in ids)
        per_step.append(same / len(ids))
    return {
        "per_step": per_step,
        "mean": float(np.nanmean(per_step)) if per_step else float("nan"),
        "n_cohort": len(ids),
        "n_vanished": registry.n_vanished(),
    }


def lineage_agreement(registry) -> dict:
    """Separate metric for division-born cells: do the same t0 roots proliferate?
    Per root, both backends' born-descendant counts (from the registry)."""
    counts = registry.lineage_counts()
    roots = sorted(registry.t0_cohort())
    both = sum(1 for r in roots if counts.get(r, {}).get("cpm", 0) > 0 and counts.get(r, {}).get("vertex", 0) > 0)
    neither = sum(1 for r in roots if counts.get(r, {}).get("cpm", 0) == 0 and counts.get(r, {}).get("vertex", 0) == 0)
    n = len(roots)
    return {
        "root_agreement": (both + neither) / n if n else float("nan"),
        "n_roots": n,
        "n_born": {b: sum(c.get(b, 0) for c in counts.values()) for b in ("cpm", "vertex")},
    }


def type_counts(run) -> dict:
    """{type: np.array over steps} counting alive cells."""
    types = sorted({str(t) for f in run for t in f.get("types", {}).values()})
    out = {t: np.zeros(len(run)) for t in types}
    for k, f in enumerate(run):
        for cid, t in f.get("types", {}).items():
            if f.get("alive", {}).get(cid, False):
                out[str(t)][k] += 1
    return out


def dynamics_distance(runA, runB) -> dict:
    """RMS distance between type-count trajectories (and field integrals if present)."""
    n = min(len(runA), len(runB))
    a, b = type_counts(runA[:n]), type_counts(runB[:n])
    per_type = {t: float(np.sqrt(np.mean((a.get(t, np.zeros(n)) - b.get(t, np.zeros(n))) ** 2)))
                for t in sorted(set(a) | set(b))}
    out = {"type_counts_rms": per_type,
           "type_counts_total": float(np.sqrt(sum(v ** 2 for v in per_type.values())))}
    names = set(runA[0].get("field_integrals", {})) & set(runB[0].get("field_integrals", {})) if n else set()
    if names:
        out["field_integrals_rms"] = {
            k: float(np.sqrt(np.mean([(runA[i]["field_integrals"][k] - runB[i]["field_integrals"][k]) ** 2
                                      for i in range(n)])))
            for k in sorted(names)}
    return out


def spatial_pattern(state_map, codes, types=("I", "D")) -> dict:
    """Per target type: patch area (px), centroid (row, col), radius of gyration (px),
    and front position = farthest patch pixel from the patch centroid (px)."""
    out = {}
    for t in types:
        if t not in codes:
            out[t] = None
            continue
        rr, cc = np.nonzero(state_map == codes[t])
        if rr.size == 0:
            out[t] = None
            continue
        pts = np.stack([rr, cc], 1).astype(float)
        cen = pts.mean(0)
        d = np.linalg.norm(pts - cen, axis=1)
        out[t] = {"area": int(rr.size), "centroid": (float(cen[0]), float(cen[1])),
                  "rg": float(np.sqrt(np.mean(d ** 2))), "front": float(d.max())}
    return out


def iou(mapA, mapB, codes=None, types=None) -> float:
    """IoU of occupied pixels (or of the listed types' pixels)."""
    if types:
        sel = [codes[t] for t in types]
        a, b = np.isin(mapA, sel), np.isin(mapB, sel)
    else:
        a, b = mapA > 0, mapB > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def ssim(mapA, mapB) -> float:
    a, b = np.asarray(mapA, float), np.asarray(mapB, float)
    rng = max(a.max(), b.max()) - min(a.min(), b.min())
    win = min(7, min(a.shape) // 2 * 2 - 1)
    if win < 3:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(structural_similarity(a, b, data_range=rng if rng > 0 else 1.0, win_size=win))


def field_match(fieldA, fieldB, rtol=1e-6) -> dict:
    """Internal check: the shared field engine should give (near-)identical fields."""
    a, b = np.asarray(fieldA, float), np.asarray(fieldB, float)
    if a.shape != b.shape:
        return {"match": False, "reason": f"shape {a.shape} != {b.shape}"}
    diff = np.abs(a - b)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
    rel = float(diff.max() / scale)
    return {"match": rel <= rtol, "max_abs": float(diff.max()), "max_rel": rel,
            "rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))}


def equivalence_tost(same, cross, margin, alpha=0.05) -> dict:
    """Two One-Sided Tests (TOST) for backend EQUIVALENCE of a fate-agreement metric.

    The capstone's honest-but-weak claim was "we fail to reject a substrate
    difference". TOST turns that into a POSITIVE claim: cross-backend agreement
    is statistically EQUIVALENT to the same-backend (seed-to-seed) baseline,
    i.e. the mean agreement *lost* going cross-backend is smaller than a
    pre-declared margin ``epsilon``.

    ``same``  : agreement values between runs of the SAME backend (different
                seeds) -- the reference/no-substrate-effect spread.
    ``cross`` : agreement values between CPM and vertex runs.
    ``margin``: epsilon, the largest agreement loss we'd still call equivalent
                (absolute units of the agreement metric, e.g. 0.05).

    Equivalence is declared when the (1-2*alpha) CI of ``diff = mean(same) -
    mean(cross)`` lies entirely within ``(-margin, +margin)`` -- equivalently,
    both one-sided Welch t-tests reject at ``alpha``. Returns the diff, its CI,
    the two one-sided p-values, the (larger) TOST p-value, and ``equivalent``.
    A positive ``diff`` means cross-backend agrees LESS than same-backend.
    """
    from scipy import stats

    a = np.asarray(same, float)
    b = np.asarray(cross, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"equivalent": False, "reason": "need >=2 samples per group",
                "n_same": na, "n_cross": nb}
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    diff = ma - mb
    se = float(np.sqrt(va / na + vb / nb))
    if se == 0.0:
        equivalent = abs(diff) < margin
        return {"equivalent": bool(equivalent), "diff": diff, "margin": float(margin),
                "se": 0.0, "ci_low": diff, "ci_high": diff, "p_lower": 0.0, "p_upper": 0.0,
                "p_tost": 0.0, "n_same": na, "n_cross": nb, "mean_same": ma, "mean_cross": mb,
                "alpha": alpha}
    # Welch-Satterthwaite dof
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # H0_upper: diff >= +margin  (reject if diff significantly below +margin)
    t_upper = (diff - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    # H0_lower: diff <= -margin  (reject if diff significantly above -margin)
    t_lower = (diff + margin) / se
    p_lower = float(stats.t.sf(t_lower, df))
    p_tost = max(p_upper, p_lower)
    tcrit = float(stats.t.ppf(1 - alpha, df))         # one-sided -> (1-2a) CI
    ci_low, ci_high = diff - tcrit * se, diff + tcrit * se
    equivalent = (ci_low > -margin) and (ci_high < margin)
    return {"equivalent": bool(equivalent), "diff": float(diff), "margin": float(margin),
            "se": se, "df": float(df), "ci_low": float(ci_low), "ci_high": float(ci_high),
            "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost,
            "n_same": na, "n_cross": nb, "mean_same": ma, "mean_cross": mb, "alpha": alpha}


def per_cell_spatial_agreement(runA, runB, registry, tol_um=5.0) -> dict:
    """Spatial analogue of :func:`per_cell_agreement`: the fraction of the t0
    cohort whose centroids end within ``tol_um`` of each other between the two
    runs (final shared frame). Sensitive to substrate-driven differences in
    *where* cells end up -- the discriminating metric for mechanics-dominated
    regimes (where fate is static) and the cohort-matched form of the capstone's
    candidate occupancy-IoU signal. A cell dead/absent in either run counts as a
    spatial disagreement."""
    ids = sorted(registry.per_cell_scored_ids())
    n = min(len(runA), len(runB))
    if n == 0 or not ids:
        return {"mean": float("nan"), "n_cohort": len(ids), "tol_um": float(tol_um)}
    fa, fb = runA[n - 1], runB[n - 1]
    close = 0
    for i in ids:
        pa = (fa.get("positions") or {}).get(i)
        pb = (fb.get("positions") or {}).get(i)
        alive = (fa.get("alive") or {}).get(i, False) and (fb.get("alive") or {}).get(i, False)
        if pa is None or pb is None or not alive:
            continue  # not close -> disagreement
        if float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) <= tol_um:
            close += 1
    return {"mean": close / len(ids), "n_cohort": len(ids), "tol_um": float(tol_um)}
