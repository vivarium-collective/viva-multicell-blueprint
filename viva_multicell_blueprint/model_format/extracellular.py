"""Field extracellular space: one numpy diffusing grid per species."""
from __future__ import annotations

import numpy as np


class FieldExtracellular:
    mode = "field"

    def __init__(self, shape=(64, 64), species=(), diffusion=0.18, decay=0.01, extent=None):
        self.shape = tuple(shape)  # (ny, nx)
        self.diffusion = diffusion
        self.decay = decay
        # world extent (width, height); default 1 unit per grid cell
        self.extent = tuple(extent) if extent else (float(shape[1]), float(shape[0]))
        self.grids = {sp: np.zeros(self.shape, float) for sp in species}

    def has(self, species: str) -> bool:
        return species in self.grids

    def _idx(self, pos):
        ny, nx = self.shape
        x = int(np.clip(pos[0] / self.extent[0] * nx, 0, nx - 1))
        y = int(np.clip(pos[1] / self.extent[1] * ny, 0, ny - 1))
        return y, x

    def deposit(self, species, pos, amount):
        self.grids[species][self._idx(pos)] += amount

    def concentration_at(self, pos, species) -> float:
        return float(self.grids[species][self._idx(pos)])

    def diffuse(self):
        for sp, g in self.grids.items():
            lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
                   + np.roll(g, 1, 1) + np.roll(g, -1, 1))
            self.grids[sp] = np.clip(g + self.diffusion * lap - self.decay * g, 0, None)
