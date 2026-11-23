"""Smoldyn particle adapter: one cell's species as counts confined to its polygon."""

from __future__ import annotations

import random
from typing import Any

from ...particles import SmoldynRunner
from ...particles.smoldyn_cell import rand_in_poly
from .. import rates
from ..registry import ModelSpec
from .base import IntracellularAdapter

_DEFAULT_POLY = [[0.0, 0.0], [8.0, 0.0], [8.0, 10.0], [0.0, 10.0]]


class ParticleAdapter(IntracellularAdapter):
    def __init__(self, model_spec: ModelSpec, initial_levels: dict[str, float], seed: int = 0):
        self.model_spec = model_spec
        self._seed = int(seed)
        self._poly = [list(map(float, p)) for p in model_spec.params.get("polygon", _DEFAULT_POLY)]
        self._species = list(model_spec.source["species"])
        self._sim_spec = rates.to_particle(model_spec)
        self._step_idx = 0
        self._runner = SmoldynRunner(self._sim_spec)
        rng = random.Random(self._seed)
        self._parts: list[list] = []
        for sp in self._species:
            n = int(round(initial_levels.get(sp, 0.0)))
            self._parts += [[sp, *p] for p in rand_in_poly(self._poly, n, rng)]

    def step(self, dt: float, inputs: dict[str, Any]) -> None:
        if not self._runner.available():
            return
        if inputs.get("polygon") is not None:
            self._poly = [list(map(float, p)) for p in inputs["polygon"]]
        spec = {**self._sim_spec, "frame_dt": float(dt)}
        self._runner.spec = spec
        self._step_idx += 1
        out = self._runner.step({"c": {"poly": self._poly, "parts": self._parts}},
                                self._seed * 100003 + self._step_idx)
        self._parts = out["c"]

    def amounts(self) -> dict[str, float]:
        counts = {s: 0.0 for s in self._species}
        for sp, *_ in self._parts:
            counts[sp] = counts.get(sp, 0.0) + 1
        return counts

    def crossed(self) -> dict[str, float]:
        return {}

    def available(self) -> bool:
        return self._runner.available()
