"""Pick the parts of a user file that are safe to execute for the runtime check.

We never run the user's whole file (it usually contains a training loop).
Only these survive, with their original line numbers:

* imports of allowed packages (torch, torchvision, math, numpy, typing,
  collections, functools, dataclasses, __future__),
* module-level assignments whose value is a literal,
* function definitions (bodies only run when the model calls them),
* class definitions, keeping only methods, docstrings and literal attributes.

Decorators are dropped except a few harmless built-ins.
"""

import ast
from typing import List

ALLOWED_PACKAGES = frozenset(["torch", "torchvision", "math", "numpy", "typing", "collections", "functools", "dataclasses", "__future__"])
_SAFE_DECORATORS = frozenset(["staticmethod", "classmethod", "property", "dataclass", "no_grad", "wraps"])


def _allowed_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return all(a.name.split(".")[0] in ALLOWED_PACKAGES for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and bool(node.module) and node.module.split(".")[0] in ALLOWED_PACKAGES
    return False


def _is_literal(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, (ast.Call, ast.Attribute, ast.Subscript, ast.Lambda, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return False
    return True


def _decorator_name(d: ast.AST) -> str:
    if isinstance(d, ast.Call):
        d = d.func
    if isinstance(d, ast.Attribute):
        return d.attr
    if isinstance(d, ast.Name):
        return d.id
    return ""


def _clean_function(fn: ast.AST) -> ast.AST:
    fn.decorator_list = [d for d in fn.decorator_list if _decorator_name(d) in _SAFE_DECORATORS]  # type: ignore[attr-defined]
    return fn


def _clean_class(cls: ast.ClassDef) -> ast.ClassDef:
    body: List[ast.stmt] = []
    for st in cls.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body.append(_clean_function(st))  # type: ignore[arg-type]
        elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            body.append(st)
        elif isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None and _is_literal(st.value):
            body.append(st)
        elif isinstance(st, ast.ClassDef):
            body.append(_clean_class(st))
    if not body:
        body = [ast.Pass(lineno=cls.lineno, col_offset=cls.col_offset)]
    cls.body = body
    cls.decorator_list = [d for d in cls.decorator_list if _decorator_name(d) in _SAFE_DECORATORS]
    cls.keywords = [k for k in cls.keywords if k.arg != "metaclass"]
    return cls


def safe_source(tree: ast.Module) -> ast.Module:
    body: List[ast.stmt] = []
    for st in tree.body:
        if _allowed_import(st):
            body.append(st)
        elif isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None and _is_literal(st.value):
            body.append(st)
        elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body.append(_clean_function(st))  # type: ignore[arg-type]
        elif isinstance(st, ast.ClassDef):
            body.append(_clean_class(st))
    mod = ast.Module(body=body, type_ignores=[])
    return ast.fix_missing_locations(mod)
