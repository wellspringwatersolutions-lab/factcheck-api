"""Deterministic arithmetic. The model never supplies the number we trust."""

from __future__ import annotations

import ast
import math
import re

_EXPR = re.compile(
    r"(?P<expr>\d+(?:\.\d+)?(?:\s*[\+\-\*/\^x×]\s*\d+(?:\.\d+)?)+)\s*"
    r"(?:=|equals|is)\s*(?P<value>-?\d+(?:\.\d+)?)",
    re.I,
)
_WORDS = re.compile(
    r"(?P<a>\d+(?:\.\d+)?)\s*(?P<op>plus|minus|times|multiplied by|divided by)\s*"
    r"(?P<b>\d+(?:\.\d+)?)\s*(?:=|equals|is)\s*(?P<value>-?\d+(?:\.\d+)?)",
    re.I,
)
_OPS = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "multiplied by": "*",
    "divided by": "/",
}


class ComputeError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 12 or abs(left) > 1_000_000):
            raise ComputeError("exponent out of range")
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise ComputeError("division by zero")
        return {
            ast.Add: left + right,
            ast.Sub: left - right,
            ast.Mult: left * right,
            ast.Div: left / right,
            ast.Mod: left % right,
            ast.Pow: left ** right,
        }[type(node.op)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"sqrt", "abs"}:
        if len(node.args) != 1:
            raise ComputeError("bad function call")
        arg = _eval(node.args[0])
        if node.func.id == "sqrt":
            if arg < 0:
                raise ComputeError("sqrt of a negative")
            return math.sqrt(arg)
        return abs(arg)
    raise ComputeError("expression is not safe arithmetic")


def safe_eval(expression: str) -> float:
    cleaned = expression.replace("×", "*").replace("x", "*").replace("^", "**")
    cleaned = re.sub(r"[^0-9\.\+\-\*/%\(\)\s\*]", "", cleaned)
    tree = ast.parse(cleaned, mode="eval")
    return float(_eval(tree.body))


def recompute(text: str) -> dict | None:
    """Return a tool result when the claim states an arithmetic equality.

    The trace is the passage a Supported or Contradicted math verdict rests on.
    """
    match = _EXPR.search(text or "")
    word = _WORDS.search(text or "")
    if match:
        expression = match.group("expr")
        claimed = float(match.group("value"))
    elif word:
        expression = f"{word.group('a')} {_OPS[word.group('op').lower()]} {word.group('b')}"
        claimed = float(word.group("value"))
    else:
        return None
    try:
        actual = safe_eval(expression)
    except (ComputeError, SyntaxError, ZeroDivisionError, ValueError) as exc:
        return {
            "tool": "calculator",
            "expression": expression,
            "claimed": claimed,
            "actual": None,
            "matches": False,
            "trace": f"Could not compute `{expression}` ({exc}).",
            "error": str(exc),
        }
    matches = abs(actual - claimed) <= 1e-6 * max(1.0, abs(actual))
    shown = int(actual) if actual.is_integer() else round(actual, 6)
    claimed_shown = int(claimed) if claimed.is_integer() else claimed
    return {
        "tool": "calculator",
        "expression": expression,
        "claimed": claimed_shown,
        "actual": shown,
        "matches": matches,
        "trace": f"{expression} = {shown}",
        "error": "",
    }
