"""Smoldyn subprocess worker — a line-oriented JSON server over stdin/stdout.

Runs in a Python interpreter that has a working ``smoldyn`` (on macOS that is a 3.11
venv; on Linux the composite imports the stepper in-process and never spawns this).
Each request line advances every cell's particles one step; the reply carries the new
particle lists. Protocol (one JSON object per line):

  <-  {"cmd": "step", "step": 7, "spec": {...}|null, "cells": {cid: {"poly": [...], "parts": [...]}}}
  ->  {"ok": true, "cells": {cid: [[species,x,y], ...]}}
  <-  {"cmd": "quit"}

Imports the stepper FLAT (its own directory on sys.path) so it never triggers the
3.12-only ``viva_multicell_blueprint`` package __init__.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoldyn_cell  # noqa: E402  (flat import — see module docstring)


def main():
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            out.write(json.dumps({"ok": False, "error": f"bad json: {e}"}) + "\n"); out.flush()
            continue
        cmd = req.get("cmd")
        if cmd == "quit":
            break
        if cmd == "ping":
            out.write(json.dumps({"ok": True, "pong": True}) + "\n"); out.flush()
            continue
        if cmd != "step":
            out.write(json.dumps({"ok": False, "error": f"unknown cmd {cmd!r}"}) + "\n"); out.flush()
            continue
        spec = req.get("spec")
        step = int(req.get("step", 0))
        result = {}
        try:
            for cid, cell in (req.get("cells") or {}).items():
                poly = cell["poly"]
                parts = cell.get("parts") or []
                seed = step * 1000003 + (abs(hash(cid)) % 9973)
                result[cid] = smoldyn_cell.step_cell(poly, parts, seed=seed, spec=spec)
            out.write(json.dumps({"ok": True, "cells": result}) + "\n"); out.flush()
        except Exception as e:
            import traceback
            out.write(json.dumps({"ok": False, "error": str(e),
                                  "trace": traceback.format_exc()}) + "\n"); out.flush()


if __name__ == "__main__":
    main()
