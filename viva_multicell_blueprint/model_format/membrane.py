"""Membrane transport contract: route crossed molecules into the extracellular field."""
from __future__ import annotations

KINDS = ("reflect", "cross", "transporter")
PERMEABLE = ("cross", "transporter")


class NoExtracellularHome(Exception):
    """A species is declared cross/transporter but has no extracellular representation."""


def _validate(field, membrane_spec):
    for sp, tr in membrane_spec.items():
        if tr.kind not in KINDS:
            raise ValueError(f"species {sp!r}: unknown transport kind {tr.kind!r}")
        if tr.kind in PERMEABLE and (field is None or not field.has(sp)):
            # TODO: viva-pde-particle particle-space extracellular (field only this milestone)
            raise NoExtracellularHome(
                f"species {sp!r} is {tr.kind!r} but has no extracellular representation")


def apply(crossed, cell_pos, field, membrane_spec):
    """Deposit permeable crossed species into ``field``; return per-cell uptake inputs."""
    _validate(field, membrane_spec)
    for sp, amount in crossed.items():
        tr = membrane_spec.get(sp)
        if tr is not None and tr.kind in PERMEABLE and amount:
            field.deposit(sp, cell_pos, amount)
    return {sp: field.concentration_at(cell_pos, sp)
            for sp, tr in membrane_spec.items() if tr.kind in PERMEABLE}
