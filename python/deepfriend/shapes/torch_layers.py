"""Read ``torch.nn`` layer constructors from the AST.

Columns are AST-native (UTF-8 byte offsets); ``analyze.py`` converts them to
UTF-16 for VS Code once at the end. Lines are 0-based.
"""

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import torch_formulas as F
from .dims import Shape, is_int

Range = Tuple[int, int, int, int]

_NO_DEFAULT = object()

# Official positional parameter order and defaults (torch 2.x).
_CONV = [("in_channels", _NO_DEFAULT), ("out_channels", _NO_DEFAULT), ("kernel_size", _NO_DEFAULT), ("stride", 1), ("padding", 0), ("dilation", 1), ("groups", 1), ("bias", True), ("padding_mode", "zeros")]
_MAXPOOL = [("kernel_size", _NO_DEFAULT), ("stride", None), ("padding", 0), ("dilation", 1), ("return_indices", False), ("ceil_mode", False)]
_AVGPOOL = [("kernel_size", _NO_DEFAULT), ("stride", None), ("padding", 0), ("ceil_mode", False), ("count_include_pad", True), ("divisor_override", None)]
_BN = [("num_features", _NO_DEFAULT), ("eps", 1e-5), ("momentum", 0.1), ("affine", True), ("track_running_stats", True)]

SIGNATURES: Dict[str, List[Tuple[str, Any]]] = {
    "Linear": [("in_features", _NO_DEFAULT), ("out_features", _NO_DEFAULT), ("bias", True)],
    "Conv1d": _CONV,
    "Conv2d": _CONV,
    "MaxPool1d": _MAXPOOL,
    "MaxPool2d": _MAXPOOL,
    "AvgPool1d": _AVGPOOL[:5],
    "AvgPool2d": _AVGPOOL,
    "AdaptiveAvgPool1d": [("output_size", _NO_DEFAULT)],
    "AdaptiveAvgPool2d": [("output_size", _NO_DEFAULT)],
    "AdaptiveMaxPool1d": [("output_size", _NO_DEFAULT), ("return_indices", False)],
    "AdaptiveMaxPool2d": [("output_size", _NO_DEFAULT), ("return_indices", False)],
    "Flatten": [("start_dim", 1), ("end_dim", -1)],
    "BatchNorm1d": _BN,
    "BatchNorm2d": _BN,
    "LayerNorm": [("normalized_shape", _NO_DEFAULT), ("eps", 1e-5), ("elementwise_affine", True), ("bias", True)],
    "Embedding": [("num_embeddings", _NO_DEFAULT), ("embedding_dim", _NO_DEFAULT), ("padding_idx", None)],
    "PReLU": [("num_parameters", 1), ("init", 0.25)],
}

# Layers that never change the shape and have no parameters.
ACTIVATIONS = frozenset(
    """ReLU ReLU6 LeakyReLU ELU SELU CELU GELU SiLU Mish Sigmoid Tanh Hardtanh Hardswish
    Hardsigmoid Softplus Softsign Softmax LogSoftmax Softmin Tanhshrink Threshold LogSigmoid
    Hardshrink Softshrink RReLU GLU_NOT Dropout Dropout1d Dropout2d Dropout3d AlphaDropout
    FeatureAlphaDropout Identity""".split()
) - {"GLU_NOT"}

CONTAINERS = frozenset(["Sequential", "ModuleList"])

TORCH_NN = "torch.nn"


@dataclass
class Layer:
    kind: str  # SIGNATURES key, activation name, "Sequential"/"ModuleList", "PReLU", or "unknown"
    name: str  # class name as written, e.g. "Conv2d"
    params: Dict[str, Any]
    line: int
    col: int
    end_line: int
    end_col: int
    arg_ranges: Dict[str, Range] = field(default_factory=dict)
    children: Optional[List["Layer"]] = None
    child_names: Optional[List[str]] = None  # submodule names ("0", "1" or OrderedDict keys)


# ---- names ----------------------------------------------------------------------


def collect_aliases(tree: ast.AST) -> Dict[str, str]:
    """Map local names to dotted module paths, e.g. {"nn": "torch.nn", "L": "torch.nn.Linear"}."""
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    top = a.name.split(".")[0]
                    aliases[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                if a.name == "*":
                    continue
                aliases[a.asname or a.name] = node.module + "." + a.name
    return aliases


def dotted(node: ast.AST) -> Optional[str]:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def qualname(func: ast.AST, aliases: Dict[str, str]) -> Optional[str]:
    """Resolve a call target to its full dotted path using the file's imports."""
    d = dotted(func)
    if d is None:
        return None
    head, _, rest = d.partition(".")
    if head in aliases:
        full = aliases[head]
    elif head == "nn":  # `nn.Linear` without a visible import: assume torch.nn
        full = TORCH_NN
    else:
        full = head
    return full + ("." + rest if rest else "")


def nn_class_name(func: ast.AST, aliases: Dict[str, str]) -> Optional[str]:
    q = qualname(func, aliases)
    if q is None:
        return None
    for prefix in ("torch.nn.modules.", "torch.nn."):
        if q.startswith(prefix):
            name = q[len(prefix):].split(".")[-1]
            return name if name[:1].isupper() else None
    return None


# ---- literal evaluation -----------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def eval_const(node: ast.AST, consts: Dict[str, Any]) -> Any:
    """Evaluate a literal-ish expression; ``None`` when it is not statically known.

    ``None`` is ambiguous with a literal ``None`` argument; callers that care
    (stride=None) treat both the same way, which matches PyTorch semantics.
    """
    try:
        return _eval(node, consts)
    except _Unknown:
        return None


class _Unknown(Exception):
    pass


def _eval(node: ast.AST, consts: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, bool)) or node.value is None:
            return node.value
        raise _Unknown
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_eval(e, consts) for e in node.elts)
    if isinstance(node, ast.Name):
        if node.id in consts and consts[node.id] is not None:
            return consts[node.id]
        raise _Unknown
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, consts)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
        raise _Unknown
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _eval(node.left, consts), _eval(node.right, consts)
        if is_int(a) and is_int(b):
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and b == 0:
                raise _Unknown
            if isinstance(node.op, ast.Pow) and (b < 0 or b > 64):
                raise _Unknown
            return _BINOPS[type(node.op)](a, b)
        raise _Unknown
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int" and len(node.args) == 1:
        v = _eval(node.args[0], consts)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
        raise _Unknown
    raise _Unknown


# ---- parsing --------------------------------------------------------------------


def _range(node: ast.AST) -> Range:
    return (node.lineno - 1, node.col_offset, (node.end_lineno or node.lineno) - 1, node.end_col_offset or node.col_offset)


def _ordered_dict_items(call: ast.Call) -> Optional[Tuple[List[str], List[ast.AST]]]:
    """``nn.Sequential(OrderedDict([("name", layer), ...]))`` -> (names, layer nodes)."""
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Call):
        return None
    inner = call.args[0]
    if (dotted(inner.func) or "").split(".")[-1] != "OrderedDict" or len(inner.args) != 1:
        return None
    seq = inner.args[0]
    if not isinstance(seq, (ast.List, ast.Tuple)):
        return None
    keys: List[str] = []
    nodes: List[ast.AST] = []
    for item in seq.elts:
        if not (isinstance(item, ast.Tuple) and len(item.elts) == 2 and isinstance(item.elts[0], ast.Constant) and isinstance(item.elts[0].value, str)):
            return None
        keys.append(item.elts[0].value)
        nodes.append(item.elts[1])
    return keys, nodes


def parse_layer(call: ast.Call, consts: Dict[str, Any], aliases: Dict[str, str]) -> Optional[Layer]:
    name = nn_class_name(call.func, aliases)
    if name is None:
        return None
    line, col, end_line, end_col = _range(call)
    if name in CONTAINERS:
        elems: List[ast.AST] = list(call.args)
        keys: Optional[List[str]] = None
        if name == "ModuleList" and len(call.args) == 1 and isinstance(call.args[0], (ast.List, ast.Tuple)):
            elems = list(call.args[0].elts)
        od = _ordered_dict_items(call) if name == "Sequential" else None
        if od is not None:
            keys, elems = od
        children: List[Layer] = []
        for e in elems:
            child = parse_layer(e, consts, aliases) if isinstance(e, ast.Call) else None
            if child is None:
                r = _range(e)
                child = Layer("unknown", "?", {}, r[0], r[1], r[2], r[3])
            children.append(child)
        names = keys if keys is not None else [str(i) for i in range(len(children))]
        return Layer(name, name, {}, line, col, end_line, end_col, children=children, child_names=names)
    if name in ACTIVATIONS:
        return Layer(name, name, {}, line, col, end_line, end_col)
    sig = SIGNATURES.get(name)
    if sig is None:
        return Layer("unknown", name, {}, line, col, end_line, end_col)
    params: Dict[str, Any] = {}
    ranges: Dict[str, Range] = {}
    names = [n for n, _ in sig]
    for i, arg in enumerate(call.args):
        if isinstance(arg, ast.Starred) or i >= len(names):
            return Layer("unknown", name, {}, line, col, end_line, end_col)
        params[names[i]] = eval_const(arg, consts)
        ranges[names[i]] = _range(arg)
    for kw in call.keywords:
        if kw.arg is None:  # **kwargs
            return Layer("unknown", name, {}, line, col, end_line, end_col)
        params[kw.arg] = eval_const(kw.value, consts)
        ranges[kw.arg] = _range(kw.value)
    for n, default in sig:
        if n not in params:
            params[n] = None if default is _NO_DEFAULT else default
    return Layer(name, name, params, line, col, end_line, end_col, ranges)


# ---- inference ---------------------------------------------------------------------


def _known(*values: Any) -> bool:
    return all(v is not None for v in values)


def infer(layer: Layer, in_shape: Shape) -> Shape:
    """Output shape of one layer. Raises ShapeError on a definite error."""
    k, p = layer.kind, layer.params
    if k in ACTIVATIONS:
        return in_shape
    if k == "Sequential":
        shape = in_shape
        for c in layer.children or []:
            shape = infer(c, shape)
        return shape
    if k == "Linear":
        if p["out_features"] is None:
            return None
        if p["in_features"] is None:
            return None
        return F.linear(in_shape, p["in_features"], p["out_features"])
    if k in ("Conv1d", "Conv2d"):
        if not _known(p["in_channels"], p["out_channels"], p["kernel_size"], p["stride"], p["padding"], p["dilation"], p["groups"]):
            return None
        return F.conv(in_shape, int(k[4]), p["in_channels"], p["out_channels"], p["kernel_size"], p["stride"], p["padding"], p["dilation"], p["groups"])
    if k in ("MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d"):
        if p.get("return_indices"):
            return None
        if not _known(p["kernel_size"], p["padding"], p["ceil_mode"]) or (k.startswith("Max") and p["dilation"] is None):
            return None
        return F.pool(in_shape, int(k[-2]), "max" if k.startswith("Max") else "avg", p["kernel_size"], p["stride"], p["padding"], p.get("dilation", 1), bool(p["ceil_mode"]))
    if k.startswith("Adaptive"):
        if p.get("return_indices") or p["output_size"] is None:
            return None
        return F.adaptive_pool(in_shape, int(k[-2]), p["output_size"])
    if k == "Flatten":
        if not _known(p["start_dim"], p["end_dim"]):
            return None
        return F.flatten(in_shape, p["start_dim"], p["end_dim"])
    if k in ("BatchNorm1d", "BatchNorm2d"):
        if p["num_features"] is None:
            return None
        return F.batchnorm(in_shape, int(k[-2]), p["num_features"])
    if k == "LayerNorm":
        if p["normalized_shape"] is None:
            return None
        return F.layernorm(in_shape, p["normalized_shape"])
    if k == "Embedding":
        if p["embedding_dim"] is None:
            return None
        return F.embedding(in_shape, p["num_embeddings"], p["embedding_dim"])
    if k == "PReLU":
        return in_shape
    return None


def _prod(v: Any) -> Optional[int]:
    vals = v if isinstance(v, tuple) else (v,)
    out = 1
    for x in vals:
        if not is_int(x):
            return None
        out *= x
    return out


def param_count(layer: Layer) -> Optional[int]:
    """Number of trainable parameters, equal to ``sum(p.numel() for p in m.parameters())``."""
    k, p = layer.kind, layer.params
    if k in ACTIVATIONS or k in ("MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d", "Flatten") or k.startswith("Adaptive"):
        return 0
    if k in CONTAINERS:
        total = 0
        for c in layer.children or []:
            n = param_count(c)
            if n is None:
                return None
            total += n
        return total
    if k == "Linear":
        i, o, b = p["in_features"], p["out_features"], p["bias"]
        if not (is_int(i) and is_int(o)) or b is None:
            return None
        return i * o + (o if b else 0)
    if k in ("Conv1d", "Conv2d"):
        i, o, g, b = p["in_channels"], p["out_channels"], p["groups"], p["bias"]
        nd = int(k[4])
        ks = p["kernel_size"]
        if not (is_int(i) and is_int(o) and is_int(g)) or b is None or ks is None:
            return None
        kk = _prod(ks if isinstance(ks, tuple) and len(ks) == nd else (ks,) * nd)
        if kk is None or g < 1 or i % g:
            return None
        return o * (i // g) * kk + (o if b else 0)
    if k in ("BatchNorm1d", "BatchNorm2d"):
        c, aff = p["num_features"], p["affine"]
        if not is_int(c) or aff is None:
            return None
        return 2 * c if aff else 0
    if k == "LayerNorm":
        n = _prod(p["normalized_shape"])
        aff, bias = p["elementwise_affine"], p["bias"]
        if n is None or aff is None or bias is None:
            return None
        if not aff:
            return 0
        return n * (2 if bias else 1)
    if k == "Embedding":
        n, d = p["num_embeddings"], p["embedding_dim"]
        if not (is_int(n) and is_int(d)):
            return None
        return n * d
    if k == "PReLU":
        n = p["num_parameters"]
        return n if is_int(n) else None
    return None
