"""Stage-2 demo: the Macklin/PhysiCell grammar driving the blueprint's ABMs.

Compiles a small, GRAMMAR-SHAPED blueprint onto a mechanics backend
(cpm | vertex | munk) and then **swaps the single rules node** from the built-in
``RulesProcess`` to ``viva_grammar``'s ``BlueprintGrammarProcess`` — same slot,
same ``fates`` / ``scalar`` / ``divisions`` / ``deaths`` output ports. So the same
shared mechanism + the same one backend node are driven by the *published
cell-behavior hypothesis grammar* (Hill-function signals→behaviors) instead of the
blueprint's ad-hoc ``rate_expr`` grammar.

Why "grammar-shaped": the grammar combines rules independently per
``(cell_type, behavior)`` with no rule ordering and no ``resistance``-scalar
chaining, so this demo uses only such rules (pressure / contact → transform /
divide / die). It is **not** a replacement for the Sego flu blueprint, which
relies on ordered + resistance rules. See the grammar repo's
``docs/stage2-blueprint-integration.md``.

Requires the ``viva-grammar`` import declared in ``workspace.yaml`` (build_core
auto-chains ``viva_grammar.core.build_core`` to register the process).
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from ..blueprint.model import Blueprint
from ..compile import compile as _compile
from ..core import register_workspace_processes

GRAMMAR_RULES_ADDR = "local:BlueprintGrammarProcess"

# A grammar-shaped rule set (independent per (cell_type, behavior); no ordering /
# no resistance scalars). Signals are read from the backend readouts alone
# (pressure = local crowding; "contact with <type>" = neighbor count), so the demo
# runs without a field source. 8-column PhysiCell CSV.
#
# NOTE: this demo intentionally uses only transform (fates) + necrosis (deaths),
# not "cycle entry" (divisions). The grammar fully supports division, but the
# installed viva_cpm CPMBackend has a PRE-EXISTING bug in its divide path
# (``World.divide_cells(div_ids, True)`` passes a list where the Rust binding
# expects a float ``threshold``) that any division — grammar or a stock ``divide``
# rule — would trip. Keeping the demo division-free lets it run cleanly on CPM;
# see the stage-2 PR note.
DEMO_RULES_CSV = """\
# crowded epithelial-like cells transition to a mesenchymal-like state
epithelial,pressure,increases,transform to mesenchymal,12.0,3.0,4,0
# an epithelial cell in contact with mesenchymal neighbours is lost
epithelial,contact with mesenchymal,increases,necrosis,6.0,1.5,4,0
"""

# base/min phenotype values the CSV does not carry (per (cell_type, behavior)).
DEMO_CELL_DEFINITIONS = {
    "epithelial": {"necrosis": {"base": 0.0}, "transform to mesenchymal": {"base": 0.0}},
    "mesenchymal": {},
}

_DEMO_BLUEPRINT = {
    "domain": {"extent_um": [24, 24], "grid_resolution_um": 1, "boundary": "noflux"},
    "cell_types": [
        {"name": "epithelial", "target_size": 16},
        {"name": "mesenchymal", "target_size": 16},
    ],
    "initial_state": {
        "kind": "blocks",
        "block": {"type": "epithelial", "origin": [8, 8], "shape": [4, 4], "spacing": 2, "size": 16},
    },
}


def swap_rules_with_grammar(doc: dict, rules_csv: str, cell_definitions: dict,
                            seed: int = 0) -> dict:
    """Replace the composite's single rules node with BlueprintGrammarProcess.

    Keeps the node's input/output wiring (same stores, same four output ports);
    only the ``address`` and ``config`` change — the clean "single-slot swap".
    """
    rules = doc["state"]["rules"]
    rules["address"] = GRAMMAR_RULES_ADDR
    rules["config"] = {
        "rules_csv": rules_csv,
        "cell_definitions": cell_definitions,
        "seed": int(seed),
        "rate_mode": "per_step",  # exact parity with the built-in RulesProcess
    }
    return doc


def _register_grammar(core):
    """core_extension: register the blueprint's own processes AND viva_grammar's.

    ``register_workspace_processes`` covers the blueprint package; the grammar
    process comes from the imported ``viva_grammar`` package, which the workspace
    ``build_core`` chains — but the dashboard subprocess builds against a bare
    core, so register it explicitly here too (idempotent).
    """
    register_workspace_processes(core)
    try:
        from viva_grammar.core import build_core as grammar_build_core
        grammar_build_core(core)
    except Exception:
        pass
    return core


@composite_generator(
    name="grammar_rules_demo",
    description=(
        "Stage-2 demo: a grammar-shaped blueprint compiled onto a mechanics backend "
        "(cpm|vertex|munk) with viva-grammar's BlueprintGrammarProcess swapped into the "
        "rules slot in place of RulesProcess — the published cell-behavior grammar "
        "(pressure/contact -> transform/divide/die) driving the real ABM."
    ),
    parameters={
        "backend": {"type": "string", "default": "cpm",
                    "description": "mechanics substrate: 'cpm', 'vertex', or 'munk'"},
        "seed": {"type": "integer", "default": 0, "description": "grammar RNG seed"},
    },
    core_extensions=[_register_grammar],
    default_n_steps=60,
)
def grammar_rules_demo(core=None, backend: str = "cpm", seed: int = 0) -> dict:
    bp = Blueprint.from_dict(dict(_DEMO_BLUEPRINT))
    doc = _compile(bp, backend=backend)  # stock composite with RulesProcess in the slot
    return swap_rules_with_grammar(doc, DEMO_RULES_CSV, DEMO_CELL_DEFINITIONS, seed=seed)
