"""Bounded-particles coupling: Smoldyn particle sims confined inside tyssue cells.

See :mod:`.smoldyn_cell` (the standalone per-cell stepper), :mod:`.worker` (the 3.11
subprocess server) and :mod:`.runner` (the in-process / subprocess dispatcher).
"""
from .runner import SmoldynRunner

__all__ = ["SmoldynRunner"]
