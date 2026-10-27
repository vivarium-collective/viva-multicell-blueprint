"""Task 18: generative latent-immune-pool placement (initial_state `extra_blocks`).

The recruitment design (Task 20) activates a PRE-PLACED LATENT pool; the Sego
blueprint therefore seeds latent immune cells at the tissue margin via an
``extra_blocks`` list appended after the main epithelial block. These guard the
generator's two load-bearing invariants:

  * extra blocks are appended with CONTIGUOUS ids after the main block;
  * the ``infected_seed`` is scoped to the MAIN block only, so adding latent
    cells never changes how many epithelial cells get infected.
"""
from __future__ import annotations

from viva_multicell_blueprint.compile._shared import _generate_blocks, resolve_initial_state
from viva_multicell_blueprint.blueprint.model import Blueprint


def test_extra_blocks_append_contiguous_ids():
    init = {
        "kind": "blocks",
        "block": {"type": "H", "origin": [10, 10], "shape": [3, 3], "spacing": 5, "size": 25},
        "extra_blocks": [
            {"type": "M_latent", "origin": [8, 43], "shape": [2, 1], "spacing": 8, "size": 25},
            {"type": "NK_latent", "origin": [8, 3], "shape": [1, 1], "spacing": 8, "size": 25},
        ],
    }
    cells = _generate_blocks(init)
    assert len(cells) == 9 + 2 + 1
    assert [c["id"] for c in cells] == list(range(1, 13))        # contiguous 1..12
    assert all(c["size"] == 25 for c in cells)                   # size-25 (no dissolve)
    kinds = {c["type"] for c in cells}
    assert {"H", "M_latent", "NK_latent"} <= kinds
    # the latent cells sit at the margin, away from the 10..20 epithelial block
    latent = [c for c in cells if c["type"].endswith("_latent")]
    assert len(latent) == 3
    assert all(c["centroid"][1] >= 40 or c["centroid"][1] <= 5 for c in latent)


def test_infected_seed_scoped_to_main_block():
    """Adding latent cells must not change the infected count: the seed only draws
    from the main (epithelial) block, so the 5%-of-36 result is invariant."""
    base = {
        "kind": "blocks",
        "block": {"type": "H", "origin": [10, 10], "shape": [6, 6], "spacing": 5, "size": 25},
        "infected_seed": {"type": "I", "fraction": 0.05, "seed": 0},
    }
    with_latent = dict(base, extra_blocks=[
        {"type": "M_latent", "origin": [8, 43], "shape": [4, 1], "spacing": 8, "size": 25},
        {"type": "NK_latent", "origin": [8, 3], "shape": [3, 1], "spacing": 10, "size": 25},
    ])
    a, b = _generate_blocks(base), _generate_blocks(with_latent)
    n_i_a = sum(c["type"] == "I" for c in a)
    n_i_b = sum(c["type"] == "I" for c in b)
    assert n_i_a == n_i_b == 2                       # 5% of the 36-cell block, both ways
    assert sum(c["type"] == "I" for c in b if c["id"] > 36) == 0   # no latent cell infected
    # the main block's infected cells are identical with/without the latent pool
    assert [c["id"] for c in a if c["type"] == "I"] == [c["id"] for c in b if c["type"] == "I"]


def test_sego_blueprint_has_latent_pool():
    """The committed Sego blueprint pre-places a size-25 latent immune pool."""
    from pathlib import Path

    bp = Blueprint.load(Path(__file__).resolve().parents[1]
                        / "workspace" / "blueprints" / "influenza_sego2022.yaml")
    cells = resolve_initial_state(bp)
    latent = [c for c in cells if str(c["type"]).endswith("_latent")]
    assert latent, "Sego blueprint must pre-place a latent immune pool (Task 20 recruitment)"
    assert all(c["size"] == 25 for c in latent)                  # size-25 (size-9 dissolves)
    types = {c["type"] for c in latent}
    assert {"M_latent", "NK_latent", "CD8_latent"} <= types
    names = {c["name"] for c in bp.cell_types}
    assert {"M_latent", "NK_latent", "CD8_latent"} <= names      # declared as cell_types
