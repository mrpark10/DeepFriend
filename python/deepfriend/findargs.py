"""Locate learning-rate and batch-size arguments for the doctor's "fix code" buttons."""

import ast
from typing import Any, Dict, List, Optional

from .shapes.analyze import to_utf16_col
from .shapes.torch_layers import collect_aliases, eval_const, qualname

TORCH_LR = {"SGD": 1e-3, "Adam": 1e-3, "AdamW": 1e-3, "RMSprop": 1e-2, "Adagrad": 1e-2, "Adadelta": 1.0, "Adamax": 2e-3, "NAdam": 2e-3, "RAdam": 1e-3, "ASGD": 1e-2, "LBFGS": 1.0}
KERAS_LR = {"SGD": 1e-2, "Adam": 1e-3, "AdamW": 1e-3, "RMSprop": 1e-3, "Adagrad": 1e-3, "Adadelta": 1e-3, "Adamax": 1e-3, "Nadam": 1e-3}


def _framework(q: str) -> Optional[str]:
    if "keras" in q or q.split(".")[0] == "optimizers":
        return "keras"
    if q.startswith("torch.optim") or q.split(".")[0] == "optim":
        return "torch"
    return None


def _loc(node: ast.AST, lines: List[str], value: Any, name: str) -> Dict[str, Any]:
    line = node.lineno - 1
    text = lines[line] if line < len(lines) else ""
    loc: Dict[str, Any] = {
        "line": line,
        "col": to_utf16_col(text, node.col_offset),
        "endCol": to_utf16_col(text, node.end_col_offset or node.col_offset),
        "value": value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
        "insert": False,
        "name": name,
    }
    return loc


def _insert_point(call: ast.Call, lines: List[str], default: float, name: str) -> Dict[str, Any]:
    end_line = (call.end_lineno or call.lineno) - 1
    text = lines[end_line]
    byte_end = (call.end_col_offset or 1) - 1  # position of ')'
    col = to_utf16_col(text, byte_end)
    return {"line": end_line, "col": col, "endCol": col, "value": None, "insert": True, "hasArgs": bool(call.args or call.keywords), "defaultValue": default, "name": name}


def find_argument(text: str, kind: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    lines = text.splitlines()
    aliases = collect_aliases(tree)
    out: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        q = qualname(node.func, aliases) or ""
        name = q.split(".")[-1]
        kw = {k.arg: k for k in node.keywords if k.arg}
        if kind == "lr":
            fw = _framework(q)
            if fw == "torch" and name in TORCH_LR:
                if "lr" in kw:
                    out.append(_loc(kw["lr"].value, lines, eval_const(kw["lr"].value, {}), "lr"))
                elif len(node.args) >= 2:
                    out.append(_loc(node.args[1], lines, eval_const(node.args[1], {}), "lr"))
                else:
                    out.append(_insert_point(node, lines, TORCH_LR[name], "lr"))
            elif fw == "keras" and name in KERAS_LR:
                if "learning_rate" in kw:
                    out.append(_loc(kw["learning_rate"].value, lines, eval_const(kw["learning_rate"].value, {}), "learning_rate"))
                elif node.args:
                    out.append(_loc(node.args[0], lines, eval_const(node.args[0], {}), "learning_rate"))
                else:
                    out.append(_insert_point(node, lines, KERAS_LR[name], "learning_rate"))
        elif kind == "batch_size":
            is_loader = name == "DataLoader"
            is_fit = isinstance(node.func, ast.Attribute) and node.func.attr == "fit"
            if not (is_loader or is_fit):
                continue
            if "batch_size" in kw:
                out.append(_loc(kw["batch_size"].value, lines, eval_const(kw["batch_size"].value, {}), "batch_size"))
            elif is_loader and len(node.args) >= 2:
                out.append(_loc(node.args[1], lines, eval_const(node.args[1], {}), "batch_size"))
            else:
                out.append(_insert_point(node, lines, 1 if is_loader else 32, "batch_size"))
    out.sort(key=lambda d: (d["line"], d["col"]))
    return out
