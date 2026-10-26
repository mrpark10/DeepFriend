"""The ``# df: input (...)`` comment and input-shape suggestions."""

import ast
import re
from dataclasses import dataclass
from typing import List, Optional

from .dims import Shape

_INPUT_RE = re.compile(r"^\s*#\s*(?:df|dm)\s*:\s*input\b\s*(.+?)\s*$", re.IGNORECASE)
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class InputSpec:
    line: int  # 0-based line of the comment
    shapes: List[Shape]
    raw: str


def parse_shapes(text: str) -> Optional[List[Shape]]:
    """Parse ``(32, 3, 224, 224)`` or ``(N, 5), (N, 7)``; dims are ints or identifiers."""
    try:
        node = ast.parse("(" + text + ",)", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Tuple) or not node.elts:
        return None
    shapes: List[Shape] = []
    for elt in node.elts:
        if not isinstance(elt, ast.Tuple):
            return None
        dims = []
        for d in elt.elts:
            if isinstance(d, ast.Constant) and isinstance(d.value, int) and not isinstance(d.value, bool) and d.value > 0:
                dims.append(d.value)
            elif isinstance(d, ast.Constant) and d.value is None:
                dims.append("None")  # Keras batch dimension
            elif isinstance(d, ast.Name) and _NAME_RE.match(d.id):
                dims.append(d.id)
            else:
                return None
        shapes.append(tuple(dims))
    return shapes


def find_input(lines: List[str], def_line: int) -> Optional[InputSpec]:
    """Find the input comment in the comment/blank/decorator block right above ``def_line``."""
    i = def_line - 1
    while i >= 0:
        text = lines[i]
        stripped = text.strip()
        if not stripped or stripped.startswith("@"):
            i -= 1
            continue
        if not stripped.startswith("#"):
            return None
        m = _INPUT_RE.match(text)
        if m:
            shapes = parse_shapes(m.group(1))
            if shapes is None:
                return None
            return InputSpec(i, shapes, m.group(1))
        i -= 1
    return None


_SUGGESTIONS = [
    (re.compile(r"\b(?:Fashion|K|E)?MNIST\b"), "(B, 1, 28, 28)"),
    (re.compile(r"\b(?:CIFAR10|CIFAR100|SVHN)\b"), "(B, 3, 32, 32)"),
    (re.compile(r"(?:\bResize|\bRandomResizedCrop|\bCenterCrop)\(\s*\(?\s*224\b|\bImageNet\b|\bImageFolder\b"), "(B, 3, 224, 224)"),
]


def suggest_input(text: str) -> Optional[str]:
    for pattern, value in _SUGGESTIONS:
        if pattern.search(text):
            return value
    return None
