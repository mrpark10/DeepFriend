"""Shape values shared by the static analyzers.

A dimension is an ``int`` or a ``str`` naming a symbolic size such as ``"B"``
(batch). A shape is a tuple of dimensions, or ``None`` when it is unknown.
Unknown is always preferred over a guess: the extension shows ``?``.
"""

from collections import Counter
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

Dim = Union[int, str]
Shape = Optional[Tuple[Dim, ...]]


class ShapeError(Exception):
    """A definite shape error. ``code`` is a DiagCode, ``info`` its message arguments."""

    def __init__(self, code: str, **info: Any) -> None:
        super().__init__("%s %r" % (code, info))
        self.code = code
        self.info: Dict[str, Any] = info


def is_int(d: Any) -> bool:
    return isinstance(d, int) and not isinstance(d, bool)


def fmt_shape(shape: Shape) -> str:
    if shape is None:
        return "?"
    if len(shape) == 1:
        return "(%s,)" % (shape[0],)
    return "(" + ", ".join(str(d) for d in shape) + ")"


def numel(shape: Shape) -> Optional[int]:
    if shape is None:
        return None
    total = 1
    for d in shape:
        if not is_int(d):
            return None
        total *= d
    return total


def factorize(dims: Iterable[Dim]) -> Tuple[int, Counter]:
    """Split a product of dims into an integer coefficient and symbol counts."""
    coef = 1
    syms: Counter = Counter()
    for d in dims:
        if is_int(d):
            coef *= d
        else:
            syms[d] += 1
    return coef, syms


def norm_dim(dim: int, rank: int, what: str = "dim") -> int:
    """Normalize a possibly negative axis index like PyTorch does."""
    lo, hi = -rank, rank - 1
    if rank == 0:
        lo, hi = -1, 0
    if not is_int(dim) or dim < lo or dim > hi:
        raise ShapeError("invalid_arg", reason="dim_range", dim=dim, rank=rank)
    return dim % max(rank, 1)


def broadcast(a: Shape, b: Shape) -> Shape:
    if a is None or b is None:
        return None
    out = []
    for i in range(1, max(len(a), len(b)) + 1):
        x = a[-i] if i <= len(a) else 1
        y = b[-i] if i <= len(b) else 1
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        elif is_int(x) and is_int(y):
            raise ShapeError("broadcast", a=fmt_shape(a), b=fmt_shape(b))
        else:
            return None
    return tuple(reversed(out))


def as_tuple(v: Any, n: int) -> Tuple[Any, ...]:
    if isinstance(v, (tuple, list)):
        if len(v) == 1:
            return tuple(v) * n
        return tuple(v)
    return (v,) * n


def same_dims(xs: Sequence[Dim]) -> bool:
    return all(x == xs[0] for x in xs)
