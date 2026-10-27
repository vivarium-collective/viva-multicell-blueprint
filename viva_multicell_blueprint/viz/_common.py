"""Shared viz helpers: decimation, rounding, colorblind-safe state colors."""
from __future__ import annotations

import numpy as np

MAX_FRAMES = 28          # mirrors viva-lumenoid _MOVIE_MAX_FRAMES
HTML_CAP = 2_500_000     # workbench _HTML_PER_FILE_MAX (bytes)
BACKEND_COLORS = {"cpm": "#0072B2", "vertex": "#D55E00"}   # Okabe-Ito blue / vermillion

# Okabe-Ito based; H blue, I orange, D dark grey, immune green.
STATE_COLORS = {"H": "#0072B2", "I": "#E69F00", "D": "#444444", "immune": "#009E73"}
_FALLBACK = ["#CC79A7", "#56B4E9", "#F0E442", "#999999"]


def state_color(t, k=0):
    return STATE_COLORS.get(t, _FALLBACK[k % len(_FALLBACK)])


def decimate_indices(n, max_n=MAX_FRAMES):
    """Evenly sampled indices, always including first and last."""
    if n <= max_n:
        return list(range(n))
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, max_n)})


def rnd(a, nd=3):
    return np.round(np.asarray(a, float), nd)


def times(n, dt_days, idx):
    return [round(i * dt_days, 4) for i in idx]


def discrete_colorscale(codes, colors):
    """Stepped colorscale for int8 maps over 0..K (zmin=0, zmax=K); 0 = transparent."""
    k = len(codes)
    cs = [[0, "rgba(0,0,0,0)"], [0.5 / k, "rgba(0,0,0,0)"]]
    for t, c in codes.items():
        cs += [[(c - 0.5) / k, colors[t]], [min(1.0, (c + 0.5) / k), colors[t]]]
    return cs
