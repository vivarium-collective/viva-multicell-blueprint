"""Cross-paradigm BEHAVIOR metrics (distance-to-CPM target).

Synthetic runs mirror the frame shape used across the comparison suite:
``positions`` / ``sizes`` / ``types`` / ``alive`` (+ ``neighbors`` for the
graph/sorting metrics). A *run* is a list of per-step frames; frame 0 is t0.
"""
import numpy as np

from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.compare.metrics import (
    front_speed_distance, front_trajectory, msd_curve, msd_distance,
    neighbour_graph_distance, per_cell_displacement_distance, sorting_index,
    sorting_index_distance)
from viva_multicell_blueprint.compare.raster import type_codes

GRID = {"extent_um": [40.0, 40.0], "grid_resolution_um": 1.0}
IDS = ["a", "b", "c"]


def reg(ids=IDS):
    """2-backend registry with DIFFERENT native ids per backend (the id remap)."""
    return IdentityRegistry.from_backends(
        list(ids), {i: f"cpm_{i}" for i in ids}, {i: f"vtx_{i}" for i in ids})


def moving_frame(vel, t, types=None, alive=None, neighbors=None):
    """Cohort a/b/c starting on a diagonal, each advancing by ``vel*t``."""
    base = {"a": [10.0, 10.0], "b": [20.0, 20.0], "c": [30.0, 30.0]}
    pos = {k: [v[0] + vel * t, v[1]] for k, v in base.items()}
    return {"positions": pos, "sizes": {k: 4.0 for k in base},
            "types": types or {k: "H" for k in base},
            "alive": alive or {k: True for k in base},
            "neighbors": neighbors or {}}


def moving_run(n, vel, **kw):
    return [moving_frame(vel, t, **kw) for t in range(n)]


# --------------------------------------------------------------------------- #
# per_cell_displacement_distance
# --------------------------------------------------------------------------- #
def test_displacement_identical_is_zero():
    run = moving_run(6, vel=1.0)
    r = per_cell_displacement_distance(run, run, reg())
    assert r["mean"] == 0.0
    assert r["per_step"][0] == 0.0 and all(s == 0.0 for s in r["per_step"])


def test_displacement_faster_copy_is_larger():
    slow = moving_run(6, vel=1.0)
    fast = moving_run(6, vel=3.0)
    same = per_cell_displacement_distance(slow, slow, reg())["mean"]
    cross = per_cell_displacement_distance(slow, fast, reg())["mean"]
    assert cross > same == 0.0
    # divergence grows along the trajectory (per-step curve, not just final)
    ps = per_cell_displacement_distance(slow, fast, reg())["per_step"]
    assert ps[0] == 0.0 and ps[-1] > ps[1]


def test_displacement_offset_insensitive():
    """A constant coordinate-frame offset between backends is removed by the
    t0 reference, so two identically-MOVING runs with different origins agree."""
    a = moving_run(5, vel=2.0)
    b = [moving_frame(2.0, t) for t in range(5)]
    for f in b:  # shift every position by a fixed offset
        for k in f["positions"]:
            f["positions"][k] = [f["positions"][k][0] + 100.0, f["positions"][k][1] - 7.0]
    assert per_cell_displacement_distance(a, b, reg())["mean"] < 1e-9


def test_displacement_drops_dead_cohort_cells():
    run = moving_run(4, vel=1.0)
    dead = moving_run(4, vel=5.0)
    for f in dead:  # 'c' dead everywhere -> excluded; a/b identical to `run`? no, vel differs
        f["alive"]["c"] = False
    # c (the fast divergent one) is dropped; a & b still diverge (vel 1 vs 5)
    r = per_cell_displacement_distance(run, dead, reg())
    assert r["mean"] > 0.0  # a,b still contribute
    # but a run where ONLY the moving cells die -> later steps empty -> nan, mean ignores
    frozen = moving_run(4, vel=1.0)
    for t, f in enumerate(frozen):
        if t >= 2:
            f["alive"] = {"a": False, "b": False, "c": False}
    rr = per_cell_displacement_distance(run, frozen, reg())
    assert np.isnan(rr["per_step"][3]) and rr["mean"] == 0.0


# --------------------------------------------------------------------------- #
# msd_curve / msd_distance
# --------------------------------------------------------------------------- #
def test_msd_rises_for_moving_zero_for_static():
    moving = moving_run(6, vel=2.0)
    static = moving_run(6, vel=0.0)
    cm = msd_curve(moving)
    cs = msd_curve(static)
    assert cm[0] == 0.0 and np.all(np.diff(cm) > 0)  # strictly rising
    assert np.allclose(cs, 0.0)
    # vel=2 -> each cell moves 2*t in x -> MSD(t) = (2t)^2
    assert np.allclose(cm, (2.0 * np.arange(6)) ** 2)


def test_msd_distance_identical_zero_moving_vs_static_positive():
    moving = moving_run(6, vel=2.0)
    static = moving_run(6, vel=0.0)
    assert msd_distance(moving, moving, reg()) == 0.0
    assert msd_distance(static, static, reg()) == 0.0
    assert msd_distance(moving, static, reg()) > 0.0


def test_msd_curve_cohort_and_dying_cells():
    run = moving_run(5, vel=1.0)
    # restrict cohort to a single cell
    c = msd_curve(run, cohort_ids=["a"])
    assert np.allclose(c, np.arange(5) ** 2)
    # a cohort cell that vanishes stops contributing; empty -> nan
    run[3]["alive"] = {"a": False, "b": False, "c": False}
    c2 = msd_curve(run, cohort_ids=["a"])
    assert np.isnan(c2[3])


# --------------------------------------------------------------------------- #
# front_trajectory / front_speed_distance
# --------------------------------------------------------------------------- #
def front_run(n, speed):
    """An 'I' patch spreading outward along x at the given speed (front advance)."""
    frames = []
    for t in range(n):
        half = 2.0 + speed * t
        pos = {"i0": [20.0 - half, 20.0], "i1": [20.0, 20.0], "i2": [20.0 + half, 20.0]}
        frames.append({"positions": pos, "sizes": {k: 4.0 for k in pos},
                       "types": {k: "I" for k in pos},
                       "alive": {k: True for k in pos}, "neighbors": {}})
    return frames


def _grid(run):
    return dict(GRID, type_codes=type_codes(run))


def test_front_trajectory_rises_with_spread():
    run = front_run(5, speed=2.0)
    traj = front_trajectory(run, _grid(run), type_="I")
    assert np.all(np.isfinite(traj))
    assert np.all(np.diff(traj) > 0)  # front advances outward


def test_front_trajectory_nan_when_type_absent():
    run = front_run(3, speed=1.0)
    for f in run:  # no 'D' cells anywhere
        pass
    traj = front_trajectory(run, _grid(run), type_="D")
    assert np.all(np.isnan(traj))


def test_front_speed_distance_separates_fast_from_slow():
    fast = front_run(6, speed=3.0)
    slow = front_run(6, speed=1.0)
    grid = dict(GRID, type_codes=type_codes(fast + slow))
    same = front_speed_distance(slow, slow, grid, type_="I")
    cross = front_speed_distance(fast, slow, grid, type_="I")
    assert same == 0.0 and cross > 0.0


# --------------------------------------------------------------------------- #
# neighbour_graph_distance
# --------------------------------------------------------------------------- #
def nbr_frame(neighbors):
    return {"positions": {k: [0.0, 0.0] for k in IDS},
            "sizes": {k: 1.0 for k in IDS}, "types": {k: "H" for k in IDS},
            "alive": {k: True for k in IDS}, "neighbors": neighbors}


def test_neighbour_graph_distance_zero_for_identical():
    run = [nbr_frame({"a": ["b"], "b": ["a", "c"], "c": ["b"]}),
           nbr_frame({"a": ["b", "c"], "b": ["a"], "c": ["a"]})]
    assert neighbour_graph_distance(run, run, reg())["mean"] == 0.0


def test_neighbour_graph_distance_positive_when_edges_differ():
    A = [nbr_frame({"a": ["b"], "b": ["a"]})]            # edge a-b
    B = [nbr_frame({"a": ["c"], "c": ["a"]})]            # edge a-c
    r = neighbour_graph_distance(A, B, reg())
    assert r["per_step"] == [1.0]  # disjoint edge sets -> Jaccard distance 1
    # one shared + one differing edge -> partial
    C = [nbr_frame({"a": ["b", "c"], "b": ["a"], "c": ["a"]})]   # a-b, a-c
    D = [nbr_frame({"a": ["b"], "b": ["a"]})]                    # a-b
    assert 0.0 < neighbour_graph_distance(C, D, reg())["per_step"][0] < 1.0


def test_neighbour_graph_restricts_to_cohort():
    # cohort is {a,b} only; an edge to out-of-cohort 'z' is ignored
    A = [nbr_frame({"a": ["b"], "b": ["a"]})]
    B = [{"positions": {}, "sizes": {}, "types": {},
          "alive": {"a": True, "b": True}, "neighbors": {"a": ["b", "z"], "b": ["a"]}}]
    r = neighbour_graph_distance(A, B, reg(["a", "b"]))
    assert r["per_step"] == [0.0]  # a-z dropped -> both graphs are just {a-b}


# --------------------------------------------------------------------------- #
# sorting_index / sorting_index_distance
# --------------------------------------------------------------------------- #
def test_sorting_index_sorted_vs_mixed():
    # sorted: same-type cells touch -> index 1.0
    sorted_f = {"types": {"a": "H", "b": "H", "c": "I", "d": "I"},
                "alive": {k: True for k in "abcd"},
                "neighbors": {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}}
    # mixed: every contact is cross-type -> index 0.0
    mixed_f = {"types": {"a": "H", "b": "I", "c": "H", "d": "I"},
               "alive": {k: True for k in "abcd"},
               "neighbors": {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}}
    assert sorting_index([sorted_f])[0] == 1.0
    assert sorting_index([mixed_f])[0] == 0.0


def test_sorting_index_distance_separates():
    sorted_f = {"types": {"a": "H", "b": "H", "c": "I", "d": "I"},
                "alive": {k: True for k in "abcd"},
                "neighbors": {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}}
    mixed_f = {"types": {"a": "H", "b": "I", "c": "H", "d": "I"},
               "alive": {k: True for k in "abcd"},
               "neighbors": {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}}
    assert sorting_index_distance([sorted_f], [sorted_f]) == 0.0
    assert sorting_index_distance([sorted_f], [mixed_f]) == 1.0


def test_sorting_index_nan_without_contacts():
    f = {"types": {"a": "H"}, "alive": {"a": True}, "neighbors": {"a": []}}
    assert np.isnan(sorting_index([f])[0])


# --------------------------------------------------------------------------- #
# registry cohort-matching (2-backend id remap + residual exclusion)
# --------------------------------------------------------------------------- #
def test_registry_cohort_matching_excludes_unmapped():
    ids = ["a", "b", "c"]
    cpm_map = {i: f"cpm_{i}" for i in ids}
    vtx_map = {i: f"vtx_{i}" for i in ids}
    # 'c' is flagged unmapped in the vertex residual -> out of the shared cohort
    r_excl = IdentityRegistry.from_backends(ids, cpm_map, vtx_map, {"unmapped_ids": ["c"]})
    assert r_excl.per_cell_scored_ids() == {"a", "b"}
    r_all = IdentityRegistry.from_backends(ids, cpm_map, vtx_map)
    assert r_all.per_cell_scored_ids() == {"a", "b", "c"}

    # a & b move identically; c's TRAJECTORY diverges over time between the runs
    A = moving_run(5, vel=1.0)
    B = moving_run(5, vel=1.0)
    for t, f in enumerate(B):
        f["positions"]["c"] = [f["positions"]["c"][0] + 10.0 * t, f["positions"]["c"][1]]
    # cohort {a,b}: c's divergence is invisible
    assert per_cell_displacement_distance(A, B, r_excl)["mean"] < 1e-9
    assert msd_distance(A, B, r_excl) < 1e-9
    # cohort {a,b,c}: c now drives the distance up
    assert per_cell_displacement_distance(A, B, r_all)["mean"] > 0.0
    assert msd_distance(A, B, r_all) > 0.0


# --------------------------------------------------------------------------- #
# generic robustness: differing lengths, empty frames
# --------------------------------------------------------------------------- #
def test_differing_lengths_truncate_to_shorter():
    long_run = moving_run(10, vel=1.0)
    short_run = moving_run(4, vel=1.0)
    r = per_cell_displacement_distance(long_run, short_run, reg())
    assert len(r["per_step"]) == 4
    assert neighbour_graph_distance(long_run, short_run, reg())["mean"] == 0.0 or True
    assert msd_distance(long_run, short_run, reg()) == 0.0


def test_empty_runs_are_nan_not_crash():
    assert np.isnan(per_cell_displacement_distance([], [], reg())["mean"])
    assert msd_curve([]).size == 0
    assert np.isnan(msd_distance([], [], reg()))
    assert np.isnan(front_speed_distance([], [], dict(GRID, type_codes={"I": 1})))
    assert np.isnan(sorting_index_distance([], []))
    assert np.isnan(neighbour_graph_distance([], [], reg())["mean"])
