"""SmoldynRunner — advance per-cell particle sims from the (3.12) composite.

Picks the best available path at construction:

1. **in-process** — ``import smoldyn`` works here (Linux/Python 3.12 has a binary
   wheel): call the stepper directly, no subprocess.
2. **subprocess** — no in-process Smoldyn (macOS/3.12): spawn :mod:`.worker` under a
   Python that *does* have Smoldyn, resolved in order: ``$SMOLDYN_PYTHON``, then an
   auto-detected sibling ``~/code/viva-smoldyn/.venv`` (local dev convenience).
3. **unavailable** — nothing found; ``available()`` is False and the caller degrades
   gracefully (the composite still runs the tissue, just with no particles).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from . import smoldyn_cell


class SmoldynRunner:
    def __init__(self, spec=None):
        self.spec = spec
        self._proc = None
        self._mode = "unavailable"
        self._python = None
        if smoldyn_cell.smoldyn_available():
            self._mode = "inproc"
            return
        py = self._resolve_python()
        if py:
            self._python = py
            self._start_worker(py)
            self._mode = "subproc"

    # -- interpreter resolution ------------------------------------------------
    @staticmethod
    def _py_has_smoldyn(py):
        try:
            r = subprocess.run([py, "-c", "import smoldyn"],
                               capture_output=True, timeout=60)
            return r.returncode == 0
        except Exception:
            return False

    def _resolve_python(self):
        env = os.environ.get("SMOLDYN_PYTHON")
        if env and Path(env).exists() and self._py_has_smoldyn(env):
            return env
        for cand in (Path.home() / "code" / "viva-smoldyn" / ".venv" / "bin" / "python",):
            if cand.exists() and self._py_has_smoldyn(str(cand)):
                return str(cand)
        return None

    def _start_worker(self, py):
        worker = str(Path(__file__).with_name("worker.py"))
        self._proc = subprocess.Popen(
            [py, worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)

    # -- API -------------------------------------------------------------------
    @property
    def mode(self):
        return self._mode

    def available(self):
        return self._mode in ("inproc", "subproc")

    def step(self, cells, step_idx):
        """cells: {cid: {"poly": [...], "parts": [[sp,x,y],...]}} -> {cid: new parts}."""
        if self._mode == "inproc":
            out = {}
            for cid, c in cells.items():
                seed = step_idx * 1000003 + (abs(hash(cid)) % 9973)
                out[cid] = smoldyn_cell.step_cell(c["poly"], c.get("parts") or [],
                                                  seed=seed, spec=self.spec)
            return out
        if self._mode == "subproc":
            req = {"cmd": "step", "step": int(step_idx), "spec": self.spec, "cells": cells}
            self._proc.stdin.write(json.dumps(req) + "\n"); self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("smoldyn worker closed unexpectedly")
            resp = json.loads(line)
            if not resp.get("ok"):
                raise RuntimeError(f"smoldyn worker: {resp.get('error')}\n{resp.get('trace','')}")
            return resp["cells"]
        raise RuntimeError("Smoldyn is not available (no in-process module and no "
                           "resolvable interpreter; set SMOLDYN_PYTHON)")

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
