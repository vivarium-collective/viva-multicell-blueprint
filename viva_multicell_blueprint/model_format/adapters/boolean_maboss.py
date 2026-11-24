"""MaBoSS Boolean-network intracellular adapter.

Uses the in-process ``cmaboss`` engine (shipped with pyMaBoSS), so no external
MaBoSS binary is needed. Each node is a stochastic ON/OFF switch with unit
rates; ``amounts()`` reports the ON-probability of every node in [0, 1].
"""

from __future__ import annotations

from typing import Any

from ..registry import ModelSpec
from .base import IntracellularAdapter

try:  # guarded: a missing engine must not break import or construction
    import cmaboss as _cmaboss

    _AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    _cmaboss = None
    _AVAILABLE = False

_SAMPLES = 400


def parse_bnet(text: str) -> dict[str, str]:
    """Parse ``node, formula`` lines (``#`` comments, ``targets`` header ok)."""
    net: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "," not in line:
            continue
        node, formula = (s.strip() for s in line.split(",", 1))
        if node.lower() == "targets":
            continue
        net[node] = formula
    return net


class BooleanAdapter(IntracellularAdapter):
    """Boolean network stepped by MaBoSS; input nodes pinned via ``inputs``."""

    def __init__(self, model_spec: ModelSpec, initial_levels: dict[str, float]):
        self.model_spec = model_spec
        src = model_spec.source
        self._net = parse_bnet(src) if isinstance(src, str) else dict(src)
        self._levels = {n: float(initial_levels.get(n, 0.0)) for n in self._net}
        self._seed = int(model_spec.params.get("seed", 1))

    def available(self) -> bool:
        return _AVAILABLE

    def crossed(self) -> dict[str, float]:
        return {}

    def amounts(self) -> dict[str, float]:
        return dict(self._levels)

    def _pins(self, inputs: dict[str, Any]) -> dict[str, float]:
        return {
            n: min(1.0, max(0.0, float(v)))
            for n, v in (inputs or {}).items()
            if n in self._net and isinstance(v, (int, float))
        }

    def step(self, dt: float, inputs: dict[str, Any]) -> None:
        pins = self._pins(inputs)
        if not _AVAILABLE:
            self._levels.update(pins)
            return
        logic = {n: ("1" if pins[n] >= 0.5 else "0") if n in pins else f
                 for n, f in self._net.items()}
        bnd = "".join(
            f"node {n} {{ logic = {f}; rate_up = @logic ? 1 : 0; "
            f"rate_down = @logic ? 0 : 1; }}\n" for n, f in logic.items())
        cfg = (f"time_tick={dt}; max_time={dt}; sample_count={_SAMPLES}; "
               f"discrete_time=0; use_physrandgen=0; seed_pseudorandom={self._seed}; "
               "thread_count=1;\n")
        for n in self._net:
            p = pins.get(n, self._levels[n])
            cfg += f"{n}.is_internal=0;\n"
            cfg += f"[{n}].istate = {p} [1], {1.0 - p} [0];\n"
        self._seed += 1
        res = _cmaboss.MaBoSSSim(network_str=bnd, config_str=cfg).run(True)
        probs, _, names = res.get_last_nodes_probtraj()
        self._levels = {n: float(probs[0][i]) for i, n in enumerate(names)}
        self._levels.update(pins)
