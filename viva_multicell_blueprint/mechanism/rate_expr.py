"""Closed rate-expression grammar (spec section 7, register 2).

``parse`` turns a string into a whitelisted AST once; ``evaluate`` runs the
identical AST against a per-cell ``CellContext`` on either backend.  No
``eval``/``exec``: only the node types below are accepted.

Variables: ``type``, ``size``, ``field['n']``, ``neighbors.count['T']``,
``neighbors.shared_surface['T']``, ``scalar['n']``, ``const['n']``; a bare
identifier (e.g. ``g_hv``) is shorthand for ``const['g_hv']``.
Functions: ``exp``, ``hill(x, k)`` = x/(k+x), ``min``, ``max``.
Operators: ``+ - * / **``, unary ``-``/``+``, parentheses, comparisons
(yielding 1.0/0.0).
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field as _dc_field
from typing import Any

__all__ = ["RateGrammarError", "CellContext", "Expr", "parse", "evaluate"]


class RateGrammarError(ValueError):
    """Expression is outside the closed rate grammar (or cannot be evaluated)."""


@dataclass
class CellContext:
    type: str
    size: float
    field: dict[str, float] = _dc_field(default_factory=dict)
    neighbor_count: dict[str, int] = _dc_field(default_factory=dict)
    shared_surface: dict[str, float] = _dc_field(default_factory=dict)
    scalar: dict[str, float] = _dc_field(default_factory=dict)
    const: dict[str, float] = _dc_field(default_factory=dict)


@dataclass(frozen=True)
class Expr:
    source: str
    tree: ast.expression


_BIN = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}
_UNARY = {ast.USub: lambda a: -a, ast.UAdd: lambda a: +a}
_CMP = {
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}
_FUNCS = {  # name -> (min_args, max_args or None)
    "exp": (1, 1), "hill": (2, 2), "min": (1, None), "max": (1, None),
}
_TABLES = {"field", "scalar", "const"}  # name['key']
_NEIGHBOR_TABLES = {"count", "shared_surface"}  # neighbors.<t>['key']
_RESERVED = _TABLES | {"neighbors"} | set(_FUNCS)


def _bad(msg: str) -> RateGrammarError:
    return RateGrammarError(msg)


def _key(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise _bad("table key must be a string literal")


def _check(node: ast.AST) -> None:
    """Recursively validate; raises RateGrammarError on anything off-grammar."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float, str)):
            raise _bad(f"literal {node.value!r} not allowed")
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN:
            raise _bad(f"operator {type(node.op).__name__} not allowed")
        _check(node.left); _check(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY:
            raise _bad(f"operator {type(node.op).__name__} not allowed")
        _check(node.operand)
    elif isinstance(node, ast.Compare):
        if any(type(o) not in _CMP for o in node.ops):
            raise _bad("comparison operator not allowed")
        for n in [node.left, *node.comparators]:
            _check(n)
    elif isinstance(node, ast.Name):
        if node.id.startswith("_") or node.id in _RESERVED:
            raise _bad(f"name {node.id!r} not allowed here")
    elif isinstance(node, ast.Subscript):
        _key(node.slice)
        v = node.value
        if isinstance(v, ast.Name) and v.id in _TABLES:
            return
        if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                and v.value.id == "neighbors" and v.attr in _NEIGHBOR_TABLES):
            return
        raise _bad("subscript must be field/scalar/const[...] or neighbors.count/shared_surface[...]")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise _bad("unknown function")
        if node.keywords:
            raise _bad("keyword arguments not allowed")
        lo, hi = _FUNCS[node.func.id]
        n = len(node.args)
        if n < lo or (hi is not None and n > hi):
            raise _bad(f"wrong argument count for {node.func.id}")
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise _bad("starred arguments not allowed")
            _check(a)
    else:
        raise _bad(f"{type(node).__name__} not allowed in rate expressions")


def parse(expr: str) -> Expr:
    if not isinstance(expr, str):
        raise _bad("expression must be a string")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise _bad(f"syntax error: {e.msg}") from e
    _check(tree.body)
    return Expr(expr, tree)


def _lookup(table: dict, key: str, what: str) -> Any:
    try:
        return table[key]
    except KeyError:
        raise _bad(f"unknown {what} {key!r}") from None


def _eval(node: ast.AST, ctx: CellContext) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _BIN[type(node.op)](_eval(node.left, ctx), _eval(node.right, ctx))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_eval(node.operand, ctx))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx)
        for op, c in zip(node.ops, node.comparators):
            right = _eval(c, ctx)
            if not _CMP[type(op)](left, right):
                return 0.0
            left = right
        return 1.0
    if isinstance(node, ast.Name):
        if node.id == "size":
            return ctx.size
        if node.id == "type":
            return ctx.type
        return _lookup(ctx.const, node.id, "constant")
    if isinstance(node, ast.Subscript):
        key = _key(node.slice)
        v = node.value
        if isinstance(v, ast.Name):
            return _lookup(getattr(ctx, v.id), key, v.id)
        table = ctx.neighbor_count if v.attr == "count" else ctx.shared_surface
        return _lookup(table, key, f"neighbors.{v.attr}")
    if isinstance(node, ast.Call):
        args = [_eval(a, ctx) for a in node.args]
        f = node.func.id
        if f == "exp":
            return math.exp(args[0])
        if f == "hill":
            return args[0] / (args[1] + args[0])
        return min(args) if f == "min" else max(args)
    raise _bad(f"{type(node).__name__} not allowed")  # unreachable after parse


def evaluate(expr: Expr, ctx: CellContext) -> float:
    try:
        v = _eval(expr.tree.body, ctx)
    except (ZeroDivisionError, OverflowError, TypeError) as e:
        raise _bad(f"cannot evaluate {expr.source!r}: {e}") from e
    if isinstance(v, str):
        raise _bad("expression yields a string, not a number")
    return float(v)
