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

from .raster import rasterize

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
    Per root, EVERY registered backend's born-descendant counts. A root 'agrees'
    when all backends concur on whether it proliferated (all > 0, or all == 0) --
    generalizes the 2-backend both/neither to N paradigms."""
    counts = registry.lineage_counts()
    roots = sorted(registry.t0_cohort())
    backends = tuple(registry.backends)

    def grew(r, b):
        return counts.get(r, {}).get(b, 0) > 0

    both = sum(1 for r in roots if all(grew(r, b) for b in backends))
    neither = sum(1 for r in roots if all(not grew(r, b) for b in backends))
    n = len(roots)
    return {
        "root_agreement": (both + neither) / n if n else float("nan"),
        "n_roots": n,
        "n_born": {b: sum(c.get(b, 0) for c in counts.values()) for b in backends},
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


def neighbour_exchanges(run) -> list[int]:
    """Cumulative count of neighbour exchanges per frame: contacts (unordered cell
    pairs in ``neighbors``) gained since the previous frame, summed. On the vertex
    substrate every gained contact is a T1/T3 topology change, so this shows
    ``auto_reconnect`` at work; frame 0 is 0."""
    def pairs(frame):
        return {tuple(sorted((str(a), str(b))))
                for a, nbrs in (frame.get("neighbors") or {}).items() for b in nbrs}

    out, total, prev = [], 0, None
    for frame in run:
        cur = pairs(frame)
        if prev is not None:
            total += len(cur - prev)
        out.append(total)
        prev = cur
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


def run_cluster_bootstrap_ci(cross_mat, same_cpm_mat, same_vtx_mat, *,
                             n_boot=4000, alpha=0.05, seed=0, labels=("cpm", "vtx")) -> dict:
    """Dependence-aware CI for diff = mean(same) - mean(cross), resampling RUNS
    (clusters), not run-PAIRS.

    This is a PAIRWISE test between any two paradigms A and B (the ``cpm``/``vtx``
    naming is historical); for a 3-way comparison it is called once per backend
    pair (cpm-vertex, cpm-munk, vertex-munk). ``labels=(A, B)`` sets the run-count
    key names in the result.

    The NxM seed grid yields N*M cross pairs and same-backend pairs that reuse only
    N A-trajectories + M B-trajectories, so the pairs are statistically DEPENDENT
    and a pair-level t-test (TOST) badly understates the uncertainty (reviewer
    guardrail). This resamples the A and B RUN indices with replacement and
    recomputes the diff on each bootstrap tissue, giving a cluster-robust
    percentile CI.

    ``cross_mat[i][j]``     = agreement(A_run_i, B_run_j).
    ``same_cpm_mat[i][j]``  = agreement(A_run_i, A_run_j) (diag ignored).
    ``same_vtx_mat[i][j]``  = agreement(B_run_i, B_run_j) (diag ignored).
    Returns the observed diff and the (1-2*alpha) percentile CI over run resamples.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    C = _np.asarray(cross_mat, float)
    Sc = _np.asarray(same_cpm_mat, float)
    Sv = _np.asarray(same_vtx_mat, float)
    nc, nv = C.shape

    def _same_mean(idx_c, idx_v):
        vals = []
        for a in range(len(idx_c)):
            for b in range(a + 1, len(idx_c)):
                if idx_c[a] != idx_c[b]:
                    vals.append(Sc[idx_c[a], idx_c[b]])
        for a in range(len(idx_v)):
            for b in range(a + 1, len(idx_v)):
                if idx_v[a] != idx_v[b]:
                    vals.append(Sv[idx_v[a], idx_v[b]])
        return float(_np.mean(vals)) if vals else _np.nan

    def _cross_mean(idx_c, idx_v):
        return float(_np.mean([C[i, j] for i in idx_c for j in idx_v]))

    all_c, all_v = list(range(nc)), list(range(nv))
    obs = _same_mean(all_c, all_v) - _cross_mean(all_c, all_v)
    diffs = []
    for _ in range(n_boot):
        ic = rng.integers(0, nc, nc).tolist()
        iv = rng.integers(0, nv, nv).tolist()
        d = _same_mean(ic, iv) - _cross_mean(ic, iv)
        if not _np.isnan(d):
            diffs.append(d)
    lo, hi = _np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    la, lb = labels
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": len(diffs), "alpha": alpha,
            f"n_{la}_runs": nc, f"n_{lb}_runs": nv,
            "n_cpm_runs": nc, "n_vtx_runs": nv,   # legacy aliases (always present)
            "method": "run-cluster percentile bootstrap"}


# --------------------------------------------------------------------------- #
# Cross-paradigm BEHAVIOR metrics (distance-to-CPM minimization targets).
#
# These quantify "same movement / same interaction pattern" across the CPM,
# vertex and center-based (munk) mechanics backends. Every one derives purely
# from existing frame fields (``positions`` / ``types`` / ``alive`` /
# ``neighbors``) -- no new emissions -- and every cohort-matched pair-metric is
# robust to:
#   * cohort cells that die / vanish  -> dropped from that step's cohort set,
#   * empty frames                    -> that step contributes NaN (nan-ignored),
#   * runs of differing length        -> truncated to the shorter.
# A *cohort* is the t0 shared set from the registry (``per_cell_scored_ids``):
# the two backends use different native cell ids, but a run is keyed by the
# shared blueprint id, so cohort cells are matched across backends by id.
# --------------------------------------------------------------------------- #


def _pos(frame, cid):
    """``(x, y)`` float array for a live, present cell, else ``None``."""
    if not (frame.get("alive") or {}).get(cid, False):
        return None
    p = (frame.get("positions") or {}).get(cid)
    return np.asarray(p, float) if p is not None else None


def per_cell_displacement_distance(runA, runB, registry) -> dict:
    """RMS over the shared cohort of ``|Δpos_A - Δpos_B|`` at each step, where
    ``Δpos(t) = pos(t) - pos(t0)`` is the cell's displacement from its t0 seed.

    The whole-curve generalization of :func:`per_cell_spatial_agreement` (which
    only looks at the final frame): it measures per-cell *trajectory* divergence
    between the two substrates, and, by referencing each run to its own t0, is
    insensitive to a constant offset between the two backends' coordinate frames.
    A cohort cell that is dead/absent (in either run) at a given step -- or at
    its own t0 -- is dropped from that step's RMS; a step with no usable cell is
    NaN. ``mean`` is the nan-ignored mean over steps."""
    ids = sorted(registry.per_cell_scored_ids())
    n = min(len(runA), len(runB))
    if n == 0 or not ids:
        return {"mean": float("nan"), "per_step": [], "n_cohort": len(ids)}
    a0, b0 = runA[0], runB[0]
    per_step = []
    for t in range(n):
        diffs = []
        for i in ids:
            pa0, pb0 = _pos(a0, i), _pos(b0, i)
            pat, pbt = _pos(runA[t], i), _pos(runB[t], i)
            if pa0 is None or pb0 is None or pat is None or pbt is None:
                continue
            d = (pat - pa0) - (pbt - pb0)
            diffs.append(float(d @ d))
        per_step.append(float(np.sqrt(np.mean(diffs))) if diffs else float("nan"))
    return {"mean": float(np.nanmean(per_step)) if per_step else float("nan"),
            "per_step": per_step, "n_cohort": len(ids)}


def msd_curve(run, cohort_ids=None) -> np.ndarray:
    """Mean-squared displacement vs step: ``MSD(t) = <|pos(t) - pos(t0)|^2>``
    over the cohort. ``cohort_ids`` defaults to the cells alive & present at
    frame 0; a cohort cell that dies/vanishes drops out of later means. Steps
    with no surviving cohort cell (or an empty run) are NaN."""
    if not run:
        return np.asarray([], float)
    f0 = run[0]
    if cohort_ids is None:
        cohort_ids = [c for c in (f0.get("positions") or {}) if _pos(f0, c) is not None]
    p0 = {c: _pos(f0, c) for c in cohort_ids}
    p0 = {c: v for c, v in p0.items() if v is not None}
    out = np.full(len(run), np.nan)
    for t, frame in enumerate(run):
        sq = []
        for c, origin in p0.items():
            p = _pos(frame, c)
            if p is None:
                continue
            d = p - origin
            sq.append(float(d @ d))
        if sq:
            out[t] = float(np.mean(sq))
    return out


def msd_distance(runA, runB, registry) -> float:
    """RMS distance between the two runs' cohort-matched MSD curves (the shared
    t0 cohort, truncated to the shorter run). NaN steps (no survivor on a side)
    are ignored; returns NaN if the curves never overlap."""
    ids = sorted(registry.per_cell_scored_ids())
    n = min(len(runA), len(runB))
    if n == 0 or not ids:
        return float("nan")
    a = msd_curve(runA[:n], ids)
    b = msd_curve(runB[:n], ids)
    d = a - b
    finite = np.isfinite(d)
    return float(np.sqrt(np.mean(d[finite] ** 2))) if finite.any() else float("nan")


def front_trajectory(run, grid_spec, type_="I") -> np.ndarray:
    """Per-frame patch *front* position (farthest patch pixel from the patch
    centroid, px) for one type, from :func:`spatial_pattern` on each rasterized
    frame. ``grid_spec`` is the common-grid spec passed to :func:`rasterize`
    (``extent_um`` / ``grid_resolution_um`` / ``type_codes``). Frames where the
    type's patch is absent (or empty) are NaN. This is the front-advance curve of
    an infection/immune front."""
    codes = grid_spec.get("type_codes") or {}
    out = np.full(len(run), np.nan)
    for t, frame in enumerate(run):
        pat = spatial_pattern(rasterize(frame, grid_spec), codes, types=(type_,)).get(type_)
        if pat is not None:
            out[t] = pat["front"]
    return out


def front_speed_distance(runA, runB, grid_spec, type_="I") -> float:
    """Distance between the two runs' front-advance curves for one type: the RMS
    gap between their :func:`front_trajectory` curves (truncated to the shorter
    run, NaN frames ignored). Separates a fast-advancing infection front from a
    slow one. Returns NaN if the fronts never co-exist."""
    n = min(len(runA), len(runB))
    if n == 0:
        return float("nan")
    a = front_trajectory(runA[:n], grid_spec, type_)
    b = front_trajectory(runB[:n], grid_spec, type_)
    d = a - b
    finite = np.isfinite(d)
    return float(np.sqrt(np.mean(d[finite] ** 2))) if finite.any() else float("nan")


def _cohort_edges(frame, cohort):
    """Undirected edge set of ``neighbors`` restricted to cohort cells."""
    edges = set()
    for a, nbrs in (frame.get("neighbors") or {}).items():
        a = str(a)
        if a not in cohort:
            continue
        for b in nbrs:
            b = str(b)
            if b in cohort and a != b:
                edges.add(tuple(sorted((a, b))))
    return edges


def neighbour_graph_distance(runA, runB, registry) -> dict:
    """Per-step Jaccard distance between the two runs' contact graphs, restricted
    to the shared t0 cohort. For each step the ``neighbors`` adjacency is reduced
    to undirected edges whose BOTH endpoints are cohort cells, then
    ``1 - |E_A & E_B| / |E_A | E_B|``. Two steps with no cohort edges on either
    side are distance 0 (identical: no interactions). Jaccard (not normalized
    graph-edit) is chosen because it is symmetric, bounded in ``[0, 1]`` and
    needs no node alignment beyond the id cohort. Dead cells carry no edges, so
    they drop out automatically. ``mean`` is over steps (truncated to shorter)."""
    cohort = set(registry.per_cell_scored_ids())
    n = min(len(runA), len(runB))
    per_step = []
    for t in range(n):
        ea, eb = _cohort_edges(runA[t], cohort), _cohort_edges(runB[t], cohort)
        union = ea | eb
        per_step.append(1.0 - len(ea & eb) / len(union) if union else 0.0)
    return {"mean": float(np.mean(per_step)) if per_step else float("nan"),
            "per_step": per_step, "n_cohort": len(cohort)}


def sorting_index(run) -> np.ndarray:
    """Same-type neighbour fraction over time: per frame, the mean over live
    cells (with >=1 live neighbour) of the fraction of their ``neighbors`` that
    share their type. 1.0 = fully sorted (contacts only within type), ~0 = fully
    mixed. Frames with no contacting live cell are NaN. Reads only
    ``neighbors`` + ``types`` + ``alive``."""
    out = np.full(len(run), np.nan)
    for t, frame in enumerate(run):
        types, alive = frame.get("types", {}), frame.get("alive", {})
        nbrs = frame.get("neighbors") or {}
        fracs = []
        for a, ns in nbrs.items():
            a = str(a)
            if not alive.get(a, False):
                continue
            live = [str(b) for b in ns if alive.get(str(b), False) and str(b) != a]
            if not live:
                continue
            same = sum(1 for b in live if str(types.get(b)) == str(types.get(a)))
            fracs.append(same / len(live))
        if fracs:
            out[t] = float(np.mean(fracs))
    return out


def sorting_index_distance(runA, runB) -> float:
    """RMS distance between the two runs' :func:`sorting_index` curves (truncated
    to the shorter run, NaN frames ignored). Separates a sorting tissue from a
    mixing one. Returns NaN if the curves never overlap."""
    n = min(len(runA), len(runB))
    if n == 0:
        return float("nan")
    d = sorting_index(runA[:n]) - sorting_index(runB[:n])
    finite = np.isfinite(d)
    return float(np.sqrt(np.mean(d[finite] ** 2))) if finite.any() else float("nan")
