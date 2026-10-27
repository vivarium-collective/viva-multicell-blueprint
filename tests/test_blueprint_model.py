import pytest
import yaml

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.blueprint.schema import validate

MINIMAL = {
    "domain": {"extent_um": [100, 100], "grid_resolution_um": 10, "boundary": "noflux"},
    "cell_types": [
        {"name": "H", "target_size": 20, "mechanics": {}, "mobile": False},
        {"name": "I", "target_size": 20, "mechanics": {}, "mobile": False},
    ],
    "adhesion": [{"a": "H", "b": "I", "strength": 1.0}],
    "initial_state": {"kind": "blocks", "block_um": 10, "type": "H"},
    "fields": [{"name": "virus", "D": 1.0, "decay": 0.1,
                "secretion": [{"type": "I", "rate": 0.5}]}],
    "rules": [{"type": "transition", "from": "H", "to": "I",
               "rate_expr": "0.1 * virus / (1 + virus)"}],
}


def _write(tmp_path, doc, name="bp.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc))
    return p


def test_roundtrip_minimal_blueprint(tmp_path):
    bp = Blueprint.load(_write(tmp_path, MINIMAL))
    assert [c["name"] for c in bp.cell_types] == ["H", "I"]
    assert bp.rules[0]["rate_expr"].startswith("0.1")
    assert bp.immune == [] or bp.immune is not None
    out = tmp_path / "out.yaml"
    bp.dump(out)
    assert Blueprint.load(out) == bp


def test_validate_accepts_minimal():
    assert validate(MINIMAL) == []


def test_validate_rejects_missing_domain():
    doc = {k: v for k, v in MINIMAL.items() if k != "domain"}
    errs = validate(doc)
    assert errs and any("domain" in e for e in errs)


def test_load_invalid_raises(tmp_path):
    doc = {k: v for k, v in MINIMAL.items() if k != "domain"}
    with pytest.raises(ValueError, match="domain"):
        Blueprint.load(_write(tmp_path, doc))


def test_initial_state_from_data_and_bad_kind():
    ok = dict(MINIMAL, initial_state={"from_data": {"cell_table": "cells.csv"}})
    assert validate(ok) == []
    bad = dict(MINIMAL, initial_state={"kind": "hexagons"})
    assert validate(bad)


def test_rule_requires_rate_expr_for_transition():
    bad = dict(MINIMAL, rules=[{"type": "transition", "from": "H", "to": "I"}])
    assert validate(bad)
