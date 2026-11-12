"""Per-study figures (spec s10).

Divergence + dynamics overlays are interactive Plotly figures (``go.Figure``). The
spatial-playback videos are server-rendered image flipbooks (:mod:`.flipbook`,
:class:`~.flipbook.Flipbook`) — each frame drawn to a crisp PNG and embedded in a
standalone pure-JS player — which replaced the fragile Plotly animation path."""
from .divergence import divergence
from .dynamics_overlay import dynamics_overlay
from .flipbook import single_playback, spatial_playback, spatial_playback_multi
from .study_figures import (
    adhesion_sweep, dose_response, equivalence_plot, mechanics_calibration, t0_fidelity,
)

__all__ = [
    "spatial_playback", "spatial_playback_multi", "single_playback", "dynamics_overlay",
    "divergence", "adhesion_sweep", "dose_response", "t0_fidelity", "equivalence_plot",
    "mechanics_calibration",
]
