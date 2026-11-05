"""The vertex mesh must stay valid geometry through a full Sego run.

A folded vertex mesh (an inverted or self-intersecting face, overlapping sheet
polygons) is not a cell shape -- it is a broken simulation. The calibrated active
motility is capped so this never happens; this pins it, and pins that the shared
field domain confines every cell (as the CPM lattice edge does).
"""
from __future__ import annotations

from pathlib import Path

from viva_multicell_blueprint.backends.vertex_mesh import is_simple, n_invalid_faces, signed_area
from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compare.runner import run_backend

SEGO = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "influenza_sego2022.yaml"
IMMUNE = {"M", "NK", "CD8", "M_latent", "NK_latent", "CD8_latent"}


def test_validity_helpers():
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert signed_area(square) == 1.0 and is_simple(square)
    assert signed_area(square[::-1]) == -1.0                       # inverted
    bowtie = [[0, 0], [1, 1], [1, 0], [0, 1]]
    assert not is_simple(bowtie)
    assert n_invalid_faces([square, square[::-1], bowtie]) == 2


def test_sego_vertex_mesh_stays_valid_and_confined():
    run = run_backend(Blueprint.load(SEGO), "vertex", 150)
    for k in range(0, 151, 10):
        f = run.frames[k]
        sheet = [p for c, p in f["polygons"].items() if f["types"][c] not in IMMUNE and p]
        assert n_invalid_faces(sheet) == 0, k
        for x, y in f["positions"].values():
            assert 0.0 <= x <= 50.0 and 0.0 <= y <= 50.0
