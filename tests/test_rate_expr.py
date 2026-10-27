import math

import pytest

from viva_multicell_blueprint.mechanism.rate_expr import (
    CellContext,
    RateGrammarError,
    evaluate,
    parse,
)


def ctx(**kw):
    base = dict(type="H", size=2.0, field={}, neighbor_count={},
                shared_surface={}, scalar={}, const={})
    base.update(kw)
    return CellContext(**base)


def test_infection_rate():
    c = ctx(field={"virus": 0.5}, const={"g_hv": 2.0})
    v = evaluate(parse("1 - exp(-g_hv * field['virus'])"), c)
    assert v == pytest.approx(1 - math.exp(-1.0))


def test_hill_regulation():
    c = ctx(field={"il10": 3.0}, const={"scale": 4.0, "k_l": 1.0})
    v = evaluate(parse("scale * hill(field['il10'], k_l)"), c)
    assert v == pytest.approx(4.0 * 3.0 / 4.0)


def test_kill_rate_uses_size_and_shared_surface():
    c = ctx(size=4.0, shared_surface={"CD8": 6.0}, scalar={"resist": 0.5},
            const={"g_ie": 2.0, "tot_ec": 3.0})
    e = parse("g_ie * tot_ec * neighbors.shared_surface['CD8'] * scalar['resist'] / size")
    assert evaluate(e, c) == pytest.approx(2 * 3 * 6 * 0.5 / 4)


def test_death_uses_prior_resist():
    c = ctx(scalar={"resist": 0.25}, const={"mu_i": 8.0})
    assert evaluate(parse("mu_i * (1 - scalar['resist'])"), c) == pytest.approx(6.0)


def test_neighbors_count_min_max_pow_compare():
    c = ctx(neighbor_count={"I": 3}, const={"a": 2.0})
    assert evaluate(parse("min(neighbors.count['I'], 2) + max(1, a) ** 2"), c) == 6.0
    assert evaluate(parse("size > 1"), c) == 1.0
    assert evaluate(parse("type == 'H'"), c) == 1.0


def test_missing_name_is_grammar_error():
    with pytest.raises(RateGrammarError):
        evaluate(parse("field['nope']"), ctx())


@pytest.mark.parametrize("bad", [
    "__import__('os')", "os.system('x')", "foo(1)", "exp.__class__",
    "size.real", "[1,2]", "lambda: 1", "(1).__add__(2)", "__builtins__",
    "field", "exp", "hill(1)", "exp(1, 2)", "x if y else z", "1 +",
    "neighbors.other['T']", "field[1]", "min()",
])
def test_rejects_out_of_grammar(bad):
    with pytest.raises(RateGrammarError):
        evaluate(parse(bad), ctx(const={"x": 1, "y": 1, "z": 1}))


@pytest.mark.parametrize("bad", [
    "9**9**9**9",          # int pow hang -> OverflowError -> grammar error
    "'a'*10**9",           # string * int allocation
    "type * 1000000000",   # type is a string too
    "'a'",                 # bare string literal
    "(-8)**0.5",           # complex result
    "(-8)**(1/3)",
    "1/0",
    "(" * 5000 + "1" + ")" * 5000,   # pathological nesting
    "+".join(["1"] * 5000),           # long left-deep chain
    "1" + " " * 5000,                 # over length cap
])
def test_resource_and_fail_closed(bad):
    with pytest.raises(RateGrammarError):
        evaluate(parse(bad), ctx())


def test_string_compare_still_allowed():
    assert evaluate(parse("'H' == type"), ctx()) == 1.0


def test_hand_built_expr_fails_closed():
    import ast
    from viva_multicell_blueprint.mechanism.rate_expr import Expr
    tree = ast.parse("'a'*1000000000", mode="eval")
    with pytest.raises(RateGrammarError):
        evaluate(Expr("x", tree), ctx())
    tree = ast.parse("(-8)**0.5", mode="eval")
    with pytest.raises(RateGrammarError):
        evaluate(Expr("x", tree), ctx())
