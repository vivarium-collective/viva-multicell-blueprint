"""SpatialVideoStep: record a single-backend run's per-step geometry and write a
self-contained spatial-playback HTML "video" to disk.

This is what lets the Blueprint Studio produce an on-demand spatial video of a
specific run. The composite-test-run path only exposes the final state to the
browser, so instead a recorder Step inside the compiled composite watches the same
geometry the mechanics node writes (positions / sizes / types / alive / footprints
+ one field grid), accumulates frames, and (re)writes the animation to a file under
``workspace/reports/assets/_studio_runs/`` — which the workbench serves at
``/assets/_studio_runs/<name>.html``, so the UI can open/download it after the run.

Enabled only when the compiler is given a ``video_out`` (see compile._shared); a
normal compile has no recorder and no file side effects.
"""
from __future__ import annotations

import os
from pathlib import Path

from process_bigraph import Step

from .compare.raster import frame_from_backend
from .viz import single_playback


class SpatialVideoStep(Step):
    """Config:
      out_path    absolute path of the HTML file to (over)write each tick;
      extent_um / grid_resolution_um   domain geometry for the raster;
      backend     'cpm' | 'vertex' (panel style + title);
      underlay    field name to show as a heat underlay (optional);
      max_frames  decimation cap for the animation.
    """
    config_schema = {
        "out_path": "string",
        "extent_um": {"_type": "list[float]", "_default": [50.0, 50.0]},
        "grid_resolution_um": {"_type": "float", "_default": 1.0},
        "backend": {"_type": "string", "_default": "cpm"},
        "underlay": {"_type": "string", "_default": ""},
        "max_frames": {"_type": "integer", "_default": 28},
    }

    def initialize(self, config):
        self._out = Path(config["out_path"])
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._grid = {"extent_um": list(config.get("extent_um") or [50.0, 50.0]),
                      "grid_resolution_um": float(config.get("grid_resolution_um", 1.0))}
        self._backend = config.get("backend", "cpm")
        self._underlay = config.get("underlay") or ""
        self._max_frames = int(config.get("max_frames", 28))
        self._frames: list = []

    def inputs(self):
        return {"positions": "map[node]", "sizes": "map[float]", "types": "map[string]",
                "alive": "map[boolean]", "footprints": "map[node]", "field_grid": "node"}

    def outputs(self):
        return {"n_frames": "overwrite[float]"}

    def _field_underlay(self, field_grid):
        if not field_grid:
            return None
        name = self._underlay if self._underlay in field_grid else next(iter(field_grid), None)
        return field_grid.get(name) if name else None

    def update(self, inputs):
        out = {
            "positions": dict(inputs.get("positions") or {}),
            "sizes": dict(inputs.get("sizes") or {}),
            "types": dict(inputs.get("types") or {}),
            "alive": dict(inputs.get("alive") or {}),
            "footprints": dict(inputs.get("footprints") or {}),
        }
        frame = frame_from_backend(out)
        fld = self._field_underlay(inputs.get("field_grid") or {})
        if fld is not None:
            frame["field"] = fld
        self._frames.append(frame)
        try:
            fig = single_playback(self._frames, self._grid, backend=self._backend,
                                  max_frames=self._max_frames,
                                  title=f"Spatial playback — {self._backend} ({len(self._frames)} steps)")
            tmp = self._out.with_suffix(".tmp")
            tmp.write_text(fig.to_html(include_plotlyjs="cdn"), encoding="utf-8")
            os.replace(tmp, self._out)   # atomic: never serve a half-written file
        except Exception:  # noqa: BLE001 — a render hiccup must not fail the simulation
            pass
        return {"n_frames": float(len(self._frames))}
