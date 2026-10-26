"""Static shape analysis for Keras models (Sequential and simple functional).

Keras infers weights lazily, so channel mismatches are not errors; only rank
problems, sizes shrinking below 1 and impossible reshapes are reported. Shapes
include the batch dimension, shown as ``None``. Columns are UTF-8 bytes.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import torch_formulas as TF
from .annotations import find_input
from .dims import Shape, ShapeError, fmt_shape, is_int
from .torch_layers import collect_aliases, eval_const, qualname

BATCH = "None"

_PREFIXES = ("tensorflow.python.keras.", "tensorflow.keras.", "tf_keras.", "keras.src.", "keras.")

_SIGS: Dict[str, List[Tuple[str, Any]]] = {
    "Dense": [("units", None), ("activation", None), ("use_bias", True)],
    "Conv2D": [("filters", None), ("kernel_size", None), ("strides", 1), ("padding", "valid"), ("data_format", None), ("dilation_rate", 1), ("groups", 1), ("activation", None), ("use_bias", True)],
    "MaxPooling2D": [("pool_size", 2), ("strides", None), ("padding", "valid"), ("data_format", None)],
    "AveragePooling2D": [("pool_size", 2), ("strides", None), ("padding", "valid"), ("data_format", None)],
    "GlobalAveragePooling2D": [("data_format", None), ("keepdims", False)],
    "GlobalMaxPooling2D": [("data_format", None), ("keepdims", False)],
    "Flatten": [("data_format", None)],
    "BatchNormalization": [("axis", -1)],
    "Embedding": [("input_dim", None), ("output_dim", None)],
    "Reshape": [("target_shape", None)],
    "InputLayer": [("shape", None), ("batch_size", None)],
}
_IDENTITY = frozenset("Dropout SpatialDropout2D GaussianNoise Activation ReLU LeakyReLU ELU PReLU Softmax LayerNormalization".split())
_POOL_ALIASES = {"MaxPool2D": "MaxPooling2D", "AvgPool2D": "AveragePooling2D", "GlobalAvgPool2D": "GlobalAveragePooling2D", "GlobalMaxPool2D": "GlobalMaxPooling2D"}


def keras_name(func: ast.AST, aliases: Dict[str, str]) -> Optional[str]:
    q = qualname(func, aliases)
    if q is None:
        return None
    for p in _PREFIXES:
        if q.startswith(p):
            rest = q[len(p):]
            parts = rest.split(".")
            if len(parts) == 1 or parts[0] in ("layers", "models") and len(parts) == 2:
                name = parts[-1]
                return _POOL_ALIASES.get(name, name)
            return None
    return None


def to_snake(name: str) -> str:
    s = re.sub("(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    return re.sub("([a-z])([A-Z])", r"\1_\2", s).lower()


@dataclass
class KLayer:
    name: str
    params: Dict[str, Any]
    kwargs: Dict[str, Any]
    line: int
    col: int
    end_line: int
    end_col: int


@dataclass
class KModel:
    name: str
    line: int
    end_line: int
    layers: List[Tuple[KLayer, int]] = field(default_factory=list)  # (layer, hint line)
    input_shape: Shape = None
    has_input: bool = False


def _klayer(call: ast.Call, consts: Dict[str, Any], aliases: Dict[str, str]) -> Optional[KLayer]:
    name = keras_name(call.func, aliases)
    if name is None or not name[:1].isupper():
        return None
    sig = _SIGS.get(name, [])
    params: Dict[str, Any] = {n: d for n, d in sig}
    for i, a in enumerate(call.args):
        if i < len(sig):
            params[sig[i][0]] = eval_const(a, consts)
    kwargs = {k.arg: eval_const(k.value, consts) for k in call.keywords if k.arg}
    for k, v in kwargs.items():
        params[k] = v
    return KLayer(name, params, kwargs, call.lineno - 1, call.col_offset, (call.end_lineno or call.lineno) - 1, call.end_col_offset or 0)


def _as2(v: Any) -> Optional[Tuple[int, int]]:
    if is_int(v):
        return (v, v)
    if isinstance(v, tuple) and len(v) == 2 and all(is_int(x) for x in v):
        return v  # type: ignore[return-value]
    return None


def _batched(shape: Any) -> Shape:
    if isinstance(shape, tuple) and all(is_int(d) or d is None for d in shape):
        return (BATCH,) + tuple(BATCH if d is None else d for d in shape)
    if is_int(shape):
        return (BATCH, shape)
    return None


def infer_keras(layer: KLayer, x: Shape) -> Shape:
    p = layer.params
    n = layer.name
    if x is None:
        return None
    if n in _IDENTITY or n == "BatchNormalization":
        return x
    if n == "Dense":
        if len(x) < 2:
            raise ShapeError("keras_rank", layer=n, expected="2D+", actual=len(x))
        if not is_int(p["units"]):
            return None
        return tuple(x[:-1]) + (p["units"],)
    if n in ("Conv2D", "MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D", "GlobalMaxPooling2D"):
        if len(x) < 4 or (len(x) > 4 and n != "Conv2D"):
            raise ShapeError("keras_rank", layer=n, expected="4D", actual=len(x))
        if len(x) > 4 or p.get("data_format") not in (None, "channels_last"):
            return None
        if n.startswith("Global"):
            return (x[0], 1, 1, x[3]) if p.get("keepdims") else (x[0], x[3])
        if n == "Conv2D":
            k, s, d = _as2(p["kernel_size"]), _as2(p["strides"]), _as2(p["dilation_rate"])
            filters, pad = p["filters"], p["padding"]
        else:
            k = _as2(p["pool_size"])
            s = _as2(p["strides"]) if p["strides"] is not None else k
            d, filters, pad = (1, 1), x[3], p["padding"]
        if k is None or s is None or d is None or pad not in ("valid", "same") or not is_int(filters):
            return None
        out = []
        for i in range(2):
            L = x[1 + i]
            if not is_int(L):
                return None
            if pad == "same":
                o = (L + s[i] - 1) // s[i]
            else:
                o = (L - d[i] * (k[i] - 1) - 1) // s[i] + 1
            if o < 1:
                raise ShapeError("nonpositive", layer=n)
            out.append(o)
        return (x[0], out[0], out[1], filters)
    if n == "Flatten":
        rest = x[1:]
        total = 1
        for dd in rest:
            if not is_int(dd):
                return None
            total *= dd
        return (x[0], total)
    if n == "Embedding":
        if not is_int(p["output_dim"]):
            return None
        return tuple(x) + (p["output_dim"],)
    if n == "Reshape":
        tgt = p["target_shape"]
        if not isinstance(tgt, tuple):
            return None
        out = TF.reshape(x[1:], tgt)
        return (x[0],) + tuple(out) if out is not None else None
    return None


def params_keras(layer: KLayer, x: Shape) -> Optional[int]:
    p = layer.params
    n = layer.name
    if n in _IDENTITY and n not in ("PReLU", "LayerNormalization"):
        return 0
    if n in ("MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D", "GlobalMaxPooling2D", "Flatten", "Reshape", "InputLayer"):
        return 0
    if x is None:
        return None
    last = x[-1]
    if n == "Dense":
        if not (is_int(last) and is_int(p["units"])) or p["use_bias"] is None:
            return None
        return last * p["units"] + (p["units"] if p["use_bias"] else 0)
    if n == "Conv2D":
        k, g = _as2(p["kernel_size"]), p["groups"]
        if k is None or not (is_int(last) and is_int(p["filters"]) and is_int(g)) or p["use_bias"] is None:
            return None
        return k[0] * k[1] * (last // g) * p["filters"] + (p["filters"] if p["use_bias"] else 0)
    if n == "BatchNormalization":
        return 4 * last if is_int(last) and p.get("axis", -1) == -1 else None
    if n == "LayerNormalization":
        return 2 * last if is_int(last) else None
    if n == "Embedding":
        a, b = p["input_dim"], p["output_dim"]
        return a * b if is_int(a) and is_int(b) else None
    return None


@dataclass(frozen=True)
class _Unknown:
    reason: str


class _Scope:
    def __init__(self, an: "KerasAnalyzer") -> None:
        self.an = an
        self.models: Dict[str, KModel] = {}
        self.tensors: Dict[str, Tuple[KModel, Shape]] = {}
        self.functional: Optional[KModel] = None
        self.consts: Dict[str, Any] = dict(an.consts)

    def run(self, body: List[ast.stmt]) -> None:
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _Scope(self.an).run(st.body)
                continue
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and isinstance(st.value, ast.Call):
                self.assign(st.targets[0].id, st.value, st)
            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                self.expr_call(st.value, st)
            elif isinstance(st, (ast.If, ast.With, ast.Try, ast.For)):
                self.run(list(st.body))

    def assign(self, target: str, call: ast.Call, st: ast.stmt) -> None:
        aliases = self.an.aliases
        name = keras_name(call.func, aliases)
        end_line = (st.end_lineno or st.lineno) - 1
        if name == "Sequential":
            m = KModel(target, st.lineno - 1, end_line)
            elems: List[ast.AST] = []
            if call.args and isinstance(call.args[0], (ast.List, ast.Tuple)):
                elems = list(call.args[0].elts)
            for k in call.keywords:
                if k.arg == "layers" and isinstance(k.value, (ast.List, ast.Tuple)):
                    elems = list(k.value.elts)
            for e in elems:
                if isinstance(e, ast.Call):
                    self.add_layer(m, e, (e.end_lineno or e.lineno) - 1)
            self.models[target] = m
            self.an.models.append(m)
            return
        if name in ("Input",) or (name == "InputLayer"):
            shape = self._input_shape(call)
            m = self.functional or KModel("functional", st.lineno - 1, end_line)
            if self.functional is None:
                self.functional = m
                m.has_input = True
                m.input_shape = shape
                self.an.models.append(m)
            self.tensors[target] = (m, shape)
            return
        if name == "Model":
            if self.functional is not None:
                self.functional.name = target
                self.functional.end_line = max(self.functional.end_line, end_line)
            return
        # functional layer application: x = layers.Dense(4)(x)
        if isinstance(call.func, ast.Call) and call.args and isinstance(call.args[0], ast.Name) and call.args[0].id in self.tensors:
            layer = _klayer(call.func, self.consts, aliases)
            if layer is None:
                return
            m, _ = self.tensors[call.args[0].id]
            m.layers.append((layer, end_line))
            m.end_line = max(m.end_line, end_line)
            self.tensors[target] = (m, None)
            layer.kwargs["__input_var"] = call.args[0].id
            layer.kwargs["__output_var"] = target

    def expr_call(self, call: ast.Call, st: ast.stmt) -> None:
        f = call.func
        if isinstance(f, ast.Attribute) and f.attr == "add" and isinstance(f.value, ast.Name) and f.value.id in self.models and call.args and isinstance(call.args[0], ast.Call):
            m = self.models[f.value.id]
            end_line = (st.end_lineno or st.lineno) - 1
            self.add_layer(m, call.args[0], end_line)
            m.end_line = max(m.end_line, end_line)

    def add_layer(self, m: KModel, call: ast.Call, hint_line: int) -> None:
        name = keras_name(call.func, self.an.aliases)
        if name in ("Input", "InputLayer"):
            m.input_shape = self._input_shape(call)
            m.has_input = True
            return
        layer = _klayer(call, self.consts, self.an.aliases)
        if layer is None:
            return
        if not m.layers and not m.has_input and "input_shape" in layer.kwargs:
            m.input_shape = _batched(layer.kwargs["input_shape"])
            m.has_input = True
        m.layers.append((layer, hint_line))

    def _input_shape(self, call: ast.Call) -> Shape:
        shape: Any = None
        batch = None
        for a in call.args[:1]:
            shape = eval_const(a, self.consts)
        for k in call.keywords:
            if k.arg in ("shape", "input_shape"):
                shape = eval_const(k.value, self.consts)
            elif k.arg in ("batch_size",):
                batch = eval_const(k.value, self.consts)
        out = _batched(shape)
        if out is not None and is_int(batch):
            out = (batch,) + tuple(out[1:])
        return out


class KerasAnalyzer:
    def __init__(self, tree: ast.Module, text: str) -> None:
        self.tree = tree
        self.text = text
        self.lines = text.splitlines()
        self.aliases = collect_aliases(tree)
        self.consts: Dict[str, Any] = {}
        self.models: List[KModel] = []

    def analyze(self) -> List[Dict[str, Any]]:
        if not any(v.split(".")[0] in ("keras", "tensorflow", "tf_keras") for v in self.aliases.values()):
            return []
        _Scope(self).run(self.tree.body)
        return [self._result(m) for m in self.models if m.layers]

    def _result(self, m: KModel) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "name": m.name, "kind": "keras", "line": m.line, "endLine": m.end_line,
            "input": None, "needsInput": False, "suggestedInput": None,
            "hints": [], "diagnostics": [], "blocks": [], "totalParams": None,
            "memory": None, "verified": False,
        }
        shape = m.input_shape
        if not m.has_input:
            spec = find_input(self.lines, m.line)
            if spec is not None and spec.shapes:
                shape = spec.shapes[0]
                info["input"] = {"line": spec.line, "shapes": [fmt_shape(s) for s in spec.shapes]}
            else:
                info["needsInput"] = True
        counters: Dict[str, int] = {}
        total: Optional[int] = 0
        line_vals: Dict[int, List[Tuple[int, Any]]] = {}
        errored = False
        for layer, hint_line in m.layers:
            base = to_snake(layer.name)
            idx = counters.get(base, 0)
            counters[base] = idx + 1
            bname = base if idx == 0 else "%s_%d" % (base, idx)
            out: Shape = None
            reason = "no_input" if info["needsInput"] else ("after_error" if errored else None)
            n_params = params_keras(layer, shape) if not info["needsInput"] else None
            err = False
            if shape is not None and not errored:
                try:
                    out = infer_keras(layer, shape)
                    if out is None:
                        reason = "unsupported_op:" + layer.name if layer.name not in _SIGS and layer.name not in _IDENTITY else "dynamic"
                except ShapeError as e:
                    info["diagnostics"].append({"line": layer.line, "col": layer.col, "endLine": layer.end_line, "endCol": layer.end_col, "code": e.code, "args": dict(e.info)})
                    errored, err, reason = True, True, "after_error"
            elif shape is None and reason is None:
                reason = "dynamic"
            if total is not None:
                total = None if n_params is None else total + n_params
            info["blocks"].append({"name": bname, "kind": layer.name, "line": layer.line, "outShape": fmt_shape(out), "params": n_params, "error": err})
            if not info["needsInput"] and not err:
                line_vals.setdefault(hint_line, []).append((layer.col, out if out is not None else _Unknown(reason or "unknown")))
            shape = out
        info["totalParams"] = total if not info["needsInput"] else None
        for line in sorted(line_vals):
            col, v = max(line_vals[line], key=lambda t: t[0])
            if isinstance(v, _Unknown):
                info["hints"].append({"line": line, "shape": "?", "unknownReason": v.reason})
            else:
                info["hints"].append({"line": line, "shape": fmt_shape(v)})
        return info


def analyze_keras(tree: ast.Module, text: str) -> List[Dict[str, Any]]:
    return KerasAnalyzer(tree, text).analyze()
