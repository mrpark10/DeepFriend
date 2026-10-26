"""Static shape analysis of ``nn.Module`` classes (no torch needed).

The analyzer reads ``__init__`` to learn the submodules, then walks
``forward`` statement by statement, tracking the shape of every local
variable. Whatever it cannot prove becomes *unknown* (shown as ``?``), never a
guess. Columns are UTF-8 byte offsets (converted by ``analyze.py``).
"""

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import torch_formulas as F
from .annotations import find_input, suggest_input
from .dims import Shape, ShapeError, broadcast, fmt_shape, is_int, numel
from .torch_layers import (
    Layer,
    collect_aliases,
    dotted,
    eval_const,
    infer,
    param_count,
    parse_layer,
    qualname,
)

MAX_DEPTH = 8
MAX_UNROLL = 64

# ---- values ----------------------------------------------------------------------


@dataclass(frozen=True)
class T:
    """A tensor with a known shape."""

    shape: Tuple[Any, ...]


@dataclass(frozen=True)
class I:
    """A Python int (or symbolic dim name). ``value=None`` means an unknown int."""

    value: Any


@dataclass(frozen=True)
class Seq:
    items: Tuple[Any, ...]


@dataclass(frozen=True)
class U:
    """Unknown value; ``reason`` explains why (shown in the hover)."""

    reason: str


@dataclass(frozen=True)
class Mod:
    ref: Any  # Layer or UserRef


@dataclass(frozen=True)
class Bound:
    """A bound helper method such as ``self.encode``."""

    name: str


@dataclass
class UserRef:
    """An instance of another model class defined in the same file."""

    cls: "ClassInfo"
    ctx: Optional["ModuleCtx"]
    line: int
    col: int
    end_line: int
    end_col: int


@dataclass
class ClassInfo:
    name: str
    node: ast.ClassDef
    bases: List[str]
    methods: Dict[str, ast.FunctionDef] = field(default_factory=dict)


@dataclass
class ModuleCtx:
    """What ``__init__`` produced for one instance."""

    cls: ClassInfo
    mods: Dict[str, Any]  # attr -> Layer | UserRef | None (unknown module)
    order: List[str]
    consts: Dict[str, Any]  # attr -> python value (int, tuple, bool...)
    params_extra: Optional[int]  # nn.Parameter elements (None = unknown)


# ---- recording ---------------------------------------------------------------------


class Recorder:
    def __init__(self) -> None:
        # line -> {key -> [values]} ; key orders entries on the same line
        self.lines: Dict[int, Dict[Tuple[int, int], List[Any]]] = {}
        self.diags: List[Dict[str, Any]] = []
        self._diag_keys: set = set()
        self.blocks: List[Dict[str, Any]] = []
        self.block_outputs: List[Shape] = []
        self.suppress = 0

    def record(self, line: int, key: Tuple[int, int], value: Any) -> None:
        if self.suppress:
            value = U("loop")
        self.lines.setdefault(line, {}).setdefault(key, []).append(value)

    def diag(self, d: Dict[str, Any]) -> None:
        if self.suppress:
            return
        k = (d["line"], d["col"], d["code"])
        if k in self._diag_keys:
            return
        self._diag_keys.add(k)
        self.diags.append(d)

    def block(self, name: str, layer: Layer, out: Any, error: bool) -> None:
        if self.suppress:
            return
        shape = out.shape if isinstance(out, T) else None
        self.blocks.append({
            "name": name,
            "kind": layer.name,
            "line": layer.line,
            "outShape": fmt_shape(shape),
            "params": param_count(layer),
            "error": error,
            "_hintLine": layer.end_line,
        })
        self.block_outputs.append(shape)


def _value_str(v: Any) -> Optional[str]:
    if isinstance(v, T):
        return fmt_shape(v.shape)
    if isinstance(v, U):
        return "?"
    if isinstance(v, Seq) and v.items and all(isinstance(x, T) for x in v.items):
        return ", ".join(fmt_shape(x.shape) for x in v.items)
    return None


def build_hints(rec: Recorder, lo: int, hi: int) -> List[Dict[str, Any]]:
    out = []
    for line in sorted(rec.lines):
        if not lo <= line <= hi:
            continue
        entries = rec.lines[line]
        key = max(entries)  # right-most entry on the line wins (one-line Sequential)
        values = entries[key]
        strs: List[str] = []
        reasons: List[str] = []
        for v in values:
            s = _value_str(v)
            if s is None:
                continue
            if s == "?":
                reasons.append(v.reason if isinstance(v, U) else "unknown")
            if s not in strs:
                strs.append(s)
        if not strs:
            continue
        known = [s for s in strs if s != "?"]
        hint: Dict[str, Any] = {"line": line, "shape": known[0] if known else "?"}
        if not known:
            hint["unknownReason"] = reasons[0] if reasons else "unknown"
        if len(known) > 1:
            hint["all"] = known
        out.append(hint)
    return out


# ---- the analyzer -------------------------------------------------------------------


class Analyzer:
    def __init__(self, tree: ast.Module, text: str) -> None:
        self.tree = tree
        self.text = text
        self.lines = text.splitlines()
        self.aliases = collect_aliases(tree)
        self.module_consts = _module_consts(tree)
        self.classes = self._find_models()

    # -- discovery --

    def _find_models(self) -> Dict[str, ClassInfo]:
        all_classes = {n.name: n for n in self.tree.body if isinstance(n, ast.ClassDef)}
        models: Dict[str, ClassInfo] = {}

        def is_model(name: str, seen: Tuple[str, ...] = ()) -> bool:
            node = all_classes.get(name)
            if node is None or name in seen:
                return False
            for b in node.bases:
                q = qualname(b, self.aliases) or ""
                if q in ("torch.nn.Module", "torch.nn.modules.module.Module") or q.endswith("nn.Module"):
                    return True
                if isinstance(b, ast.Name) and is_model(b.id, seen + (name,)):
                    return True
            return False

        for name, node in all_classes.items():
            if is_model(name):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name) and b.id in all_classes]
                models[name] = ClassInfo(name, node, bases)
        for info in models.values():
            info.methods = self._methods(info, models)
        return models

    def _methods(self, info: ClassInfo, models: Dict[str, ClassInfo], depth: int = 0) -> Dict[str, ast.FunctionDef]:
        methods: Dict[str, ast.FunctionDef] = {}
        if depth < 5:
            for b in info.bases:
                if b in models:
                    methods.update(self._methods(models[b], models, depth + 1))
        for n in info.node.body:
            if isinstance(n, ast.FunctionDef):
                methods[n.name] = n
        return methods

    # -- __init__ --

    def build_ctx(self, cls: ClassInfo, args: Sequence[ast.AST], kwargs: Dict[str, ast.AST], caller_consts: Dict[str, Any], depth: int = 0) -> ModuleCtx:
        init = cls.methods.get("__init__")
        consts: Dict[str, Any] = dict(self.module_consts)
        if init is not None:
            consts.update(_bind_ctor(init, args, kwargs, caller_consts, self.module_consts))
        ctx = ModuleCtx(cls, {}, [], {}, 0)
        if init is not None:
            self._exec_init(init.body, ctx, consts, depth, in_loop=False)
        return ctx

    def _exec_init(self, body: List[ast.stmt], ctx: ModuleCtx, consts: Dict[str, Any], depth: int, in_loop: bool) -> None:
        for st in body:
            if isinstance(st, (ast.Assign, ast.AnnAssign)):
                targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                value = st.value
                if value is None:
                    continue
                for tgt in targets:
                    if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                        self._assign_self(tgt.attr, value, ctx, consts, depth, in_loop)
                    elif isinstance(tgt, ast.Name):
                        consts[tgt.id] = None if in_loop else eval_const(value, consts)
            elif isinstance(st, ast.If):
                cond = _const_bool(st.test, consts, ctx.consts)
                if cond is True:
                    self._exec_init(st.body, ctx, consts, depth, in_loop)
                elif cond is False:
                    self._exec_init(st.orelse, ctx, consts, depth, in_loop)
                else:
                    self._exec_init(st.body + st.orelse, ctx, consts, depth, True)
            elif isinstance(st, (ast.For, ast.While, ast.With, ast.Try)):
                inner = list(st.body) + list(getattr(st, "orelse", []))
                self._exec_init(inner, ctx, consts, depth, True)

    def _assign_self(self, attr: str, value: ast.AST, ctx: ModuleCtx, consts: Dict[str, Any], depth: int, in_loop: bool) -> None:
        mod: Any = "not-a-module"
        if isinstance(value, ast.Call):
            layer = parse_layer(value, consts, self.aliases)
            if layer is not None:
                mod = layer
            elif isinstance(value.func, ast.Name) and value.func.id in self.classes and depth < MAX_DEPTH:
                sub = self.classes[value.func.id]
                kwargs = {k.arg: k.value for k in value.keywords if k.arg}
                sub_ctx = self.build_ctx(sub, value.args, kwargs, consts, depth + 1)
                mod = UserRef(sub, sub_ctx, value.lineno - 1, value.col_offset, (value.end_lineno or value.lineno) - 1, value.end_col_offset or 0)
            else:
                n = _parameter_numel(value, consts, self.aliases)
                if n != "no":
                    if ctx.params_extra is not None:
                        ctx.params_extra = None if n is None or in_loop else ctx.params_extra + n
                    return
        if mod == "not-a-module":
            ctx.consts[attr] = None if in_loop else eval_const(value, consts)
            return
        if in_loop:
            mod = None
        if attr not in ctx.mods:
            ctx.order.append(attr)
        ctx.mods[attr] = mod
        ctx.consts.pop(attr, None)

    # -- forward --

    def run_model(self, ctx: ModuleCtx, inputs: List[Any], rec: Recorder, depth: int, prefix: str) -> Any:
        fwd = ctx.cls.methods.get("forward")
        if fwd is None:
            return U("unsupported_op:forward")
        return self._call_function(fwd, inputs, {}, ctx, rec, depth, prefix)

    def _call_function(self, fn: ast.FunctionDef, args: List[Any], kwargs: Dict[str, Any], ctx: ModuleCtx, rec: Recorder, depth: int, prefix: str) -> Any:
        if depth > MAX_DEPTH:
            return U("unsupported_op:recursion")
        env: Dict[str, Any] = {}
        params = fn.args.args[1:]  # skip self
        defaults = fn.args.defaults
        first_default = len(params) - len(defaults)
        for i, p in enumerate(params):
            if i < len(args):
                env[p.arg] = args[i]
            elif p.arg in kwargs:
                env[p.arg] = kwargs[p.arg]
            elif i >= first_default:
                env[p.arg] = _py_to_value(eval_const(defaults[i - first_default], self.module_consts))
            else:
                env[p.arg] = U("no_input")
        frame = Frame(self, ctx, rec, depth, prefix)
        returns: List[Any] = []
        frame.exec_body(fn.body, env, returns)
        if not returns:
            return U("unknown")
        first = returns[0]
        if all(r == first for r in returns):
            return first
        return U("branch")

    # -- totals --

    def total_params(self, ctx: ModuleCtx) -> Optional[int]:
        if ctx.params_extra is None:
            return None
        total = ctx.params_extra
        for attr in ctx.order:
            m = ctx.mods[attr]
            if m is None:
                return None
            if isinstance(m, UserRef):
                n = self.total_params(m.ctx) if m.ctx else None
            else:
                n = param_count(m)
            if n is None:
                return None
            total += n
        return total

    def definition_blocks(self, ctx: ModuleCtx, prefix: str = "") -> List[Dict[str, Any]]:
        blocks = []
        for attr in ctx.order:
            m = ctx.mods[attr]
            name = prefix + attr
            if isinstance(m, Layer):
                for child_name, leaf in _leaves(m, name):
                    blocks.append({"name": child_name, "kind": leaf.name, "line": leaf.line, "outShape": "?", "params": param_count(leaf), "error": False, "_hintLine": leaf.end_line})
            elif isinstance(m, UserRef) and m.ctx is not None:
                blocks.extend(self.definition_blocks(m.ctx, name + "."))
        return blocks

    # -- entry point --

    def analyze(self) -> List[Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        used_as_sub: set = set()
        ctxs: Dict[str, ModuleCtx] = {}
        for name, cls in self.classes.items():
            try:
                ctxs[name] = self.build_ctx(cls, [], {}, {})
            except RecursionError:
                continue
            for m in ctxs[name].mods.values():
                if isinstance(m, UserRef):
                    used_as_sub.add(m.cls.name)

        for name, cls in self.classes.items():
            if name not in ctxs:
                continue
            node = cls.node
            def_line = node.lineno - 1
            spec = find_input(self.lines, def_line - len(node.decorator_list) if node.decorator_list else def_line)
            info: Dict[str, Any] = {
                "name": name,
                "kind": "torch",
                "line": def_line,
                "endLine": (node.end_lineno or node.lineno) - 1,
                "input": None,
                "needsInput": False,
                "suggestedInput": None,
                "hints": [],
                "diagnostics": [],
                "blocks": [],
                "totalParams": self.total_params(ctxs[name]),
                "memory": None,
                "verified": False,
            }
            results[name] = info
            ctx = ctxs[name]
            if spec is None:
                info["needsInput"] = name not in used_as_sub
                if info["needsInput"]:
                    info["suggestedInput"] = suggest_input(self.text)
                info["blocks"] = self.definition_blocks(ctx)
                continue
            info["input"] = {"line": spec.line, "shapes": [fmt_shape(s) for s in spec.shapes]}
            rec = Recorder()
            inputs = [T(tuple(s)) for s in spec.shapes]  # type: ignore[arg-type]
            try:
                self.run_model(ctx, inputs, rec, 0, "")
            except RecursionError:
                pass
            lo, hi = info["line"], info["endLine"]
            info["hints"] = build_hints(rec, lo, hi)
            info["blocks"] = rec.blocks
            info["memory"] = _memory(info["totalParams"], rec.block_outputs)
            # diagnostics: own lines here, others are distributed below
            for d in rec.diags:
                owner = self._owner(d["line"])
                if owner == name:
                    info["diagnostics"].append(d)
            info["_rec"] = rec

        # nested modules without their own input comment borrow the first caller's shapes
        for info in results.values():
            if info["input"] is not None or info["needsInput"]:
                continue
            for caller in results.values():
                rec = caller.get("_rec")
                if rec is None:
                    continue
                hints = build_hints(rec, info["line"], info["endLine"])
                if hints:
                    info["hints"] = hints
                    info["diagnostics"] = [d for d in rec.diags if info["line"] <= d["line"] <= info["endLine"]]
                    break
        for info in results.values():
            info.pop("_rec", None)
        return list(results.values())

    def _owner(self, line: int) -> Optional[str]:
        for name, cls in self.classes.items():
            if cls.node.lineno - 1 <= line <= (cls.node.end_lineno or cls.node.lineno) - 1:
                return name
        return None


# ---- statement execution ---------------------------------------------------------------


class Frame:
    def __init__(self, an: Analyzer, ctx: ModuleCtx, rec: Recorder, depth: int, prefix: str) -> None:
        self.an = an
        self.ctx = ctx
        self.rec = rec
        self.depth = depth
        self.prefix = prefix
        self.error_node: Optional[ast.AST] = None

    # statements

    def exec_body(self, body: List[ast.stmt], env: Dict[str, Any], returns: List[Any]) -> None:
        for st in body:
            self.exec_stmt(st, env, returns)

    def exec_stmt(self, st: ast.stmt, env: Dict[str, Any], returns: List[Any]) -> None:
        end_line = (st.end_lineno or st.lineno) - 1
        key = (st.lineno - 1, st.col_offset)
        if isinstance(st, ast.Assign):
            val = self.ev(st.value, env)
            for tgt in st.targets:
                self.assign(tgt, val, env)
            if len(st.targets) == 1:
                self._record_target(st.targets[0], val, end_line, key)
        elif isinstance(st, ast.AnnAssign) and st.value is not None:
            val = self.ev(st.value, env)
            self.assign(st.target, val, env)
            self._record_target(st.target, val, end_line, key)
        elif isinstance(st, ast.AugAssign):
            cur = self.ev(st.target, env)
            val = self._binop(cur, self.ev(st.value, env), st.op, st)
            self.assign(st.target, val, env)
            self._record_target(st.target, val, end_line, key)
        elif isinstance(st, ast.Return):
            val = self.ev(st.value, env) if st.value is not None else U("unknown")
            returns.append(val)
            s = _value_str(val)
            if s is not None:
                self.rec.record(end_line, key, val)
        elif isinstance(st, ast.If):
            cond = _const_bool(st.test, {}, self.ctx.consts, env)
            if cond is True:
                self.exec_body(st.body, env, returns)
            elif cond is False:
                self.exec_body(st.orelse, env, returns)
            else:
                env_a, env_b = dict(env), dict(env)
                self.exec_body(st.body, env_a, returns)
                self.exec_body(st.orelse, env_b, returns)
                for k in set(env_a) | set(env_b):
                    a, b = env_a.get(k, env_b.get(k)), env_b.get(k, env_a.get(k))
                    env[k] = a if a == b else U("branch")
        elif isinstance(st, ast.For):
            self.exec_for(st, env, returns)
        elif isinstance(st, ast.While):
            self.exec_unknown_loop(st.body + st.orelse, env, returns)
        elif isinstance(st, ast.With):
            self.exec_body(st.body, env, returns)
        elif isinstance(st, ast.Try):
            self.exec_body(st.body + st.orelse + st.finalbody, env, returns)

    def _record_target(self, tgt: ast.AST, val: Any, end_line: int, key: Tuple[int, int]) -> None:
        if isinstance(tgt, (ast.Name, ast.Attribute, ast.Subscript, ast.Tuple)) and _value_str(val) is not None:
            self.rec.record(end_line, key, val)

    def assign(self, tgt: ast.AST, val: Any, env: Dict[str, Any]) -> None:
        if isinstance(tgt, ast.Name):
            env[tgt.id] = val
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            n = len(tgt.elts)
            if isinstance(val, Seq) and len(val.items) == n:
                for t, v in zip(tgt.elts, val.items):
                    self.assign(t, v, env)
            else:
                reason = val.reason if isinstance(val, U) else "unknown"
                for t in tgt.elts:
                    self.assign(t, U(reason), env)
        elif isinstance(tgt, ast.Starred):
            self.assign(tgt.value, U("unknown"), env)

    def exec_for(self, st: ast.For, env: Dict[str, Any], returns: List[Any]) -> None:
        it = st.iter
        items: Optional[List[Any]] = None
        enum = False
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "enumerate" and it.args:
            enum = True
            it = it.args[0]
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and not enum:
            vals = [self.ev(a, env) for a in it.args]
            ints = [v.value if isinstance(v, I) and is_int(v.value) else None for v in vals]
            if ints and None not in ints and len(ints) <= 3:
                rng = range(*ints)
                if len(rng) <= MAX_UNROLL:
                    items = [I(i) for i in rng]
        else:
            v = self.ev(it, env)
            if isinstance(v, Mod) and isinstance(v.ref, Layer) and v.ref.kind in ("Sequential", "ModuleList"):
                items = [Mod(c) for c in v.ref.children or []]
            elif isinstance(v, Seq) and len(v.items) <= MAX_UNROLL:
                items = list(v.items)
        if items is None or (st.orelse and False):
            self.exec_unknown_loop(st.body + st.orelse, env, returns, st.target)
            return
        for i, item in enumerate(items):
            self.assign(st.target, Seq((I(i), item)) if enum else item, env)
            self.exec_body(st.body, env, returns)
        self.exec_body(st.orelse, env, returns)

    def exec_unknown_loop(self, body: List[ast.stmt], env: Dict[str, Any], returns: List[Any], target: Optional[ast.AST] = None) -> None:
        if target is not None:
            self.assign(target, U("loop"), env)
        inner = dict(env)
        self.rec.suppress += 1
        try:
            self.exec_body(body, inner, [])
        finally:
            self.rec.suppress -= 1
        for k in inner:
            if k not in env or inner[k] != env[k]:
                env[k] = U("loop")

    # expressions

    def ev(self, node: Optional[ast.AST], env: Dict[str, Any]) -> Any:
        if node is None:
            return U("unknown")
        if isinstance(node, ast.Constant):
            v = node.value
            if is_int(v):
                return I(v)
            return Seq(()) if v is None else I(v) if isinstance(v, (bool, float, str)) else U("unknown")
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self.an.module_consts and self.an.module_consts[node.id] is not None:
                return _py_to_value(self.an.module_consts[node.id])
            return U("unsupported_op:" + node.id)
        if isinstance(node, (ast.Tuple, ast.List)):
            return Seq(tuple(self.ev(e, env) for e in node.elts))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = self.ev(node.operand, env)
            if isinstance(v, I) and is_int(v.value):
                return I(-v.value)
            return v if isinstance(v, (T, U)) else I(None)
        if isinstance(node, ast.BinOp):
            return self._binop(self.ev(node.left, env), self.ev(node.right, env), node.op, node)
        if isinstance(node, ast.Attribute):
            return self.ev_attribute(node, env)
        if isinstance(node, ast.Subscript):
            return self.ev_subscript(node, env)
        if isinstance(node, ast.Call):
            return self.ev_call(node, env)
        return U("unsupported_op:" + type(node).__name__)

    def ev_attribute(self, node: ast.Attribute, env: Dict[str, Any]) -> Any:
        if isinstance(node.value, ast.Name) and node.value.id == "self" and "self" not in env:
            attr = node.attr
            if attr in self.ctx.mods:
                m = self.ctx.mods[attr]
                return Mod(m) if m is not None else U("unsupported_op:self." + attr)
            if attr in self.ctx.cls.methods:
                return Bound(attr)
            if attr in self.ctx.consts and self.ctx.consts[attr] is not None:
                return _py_to_value(self.ctx.consts[attr])
            return U("unsupported_op:self." + attr)
        base = self.ev(node.value, env)
        if isinstance(base, T):
            if node.attr == "shape":
                return Seq(tuple(I(d) for d in base.shape))
            if node.attr == "ndim":
                return I(len(base.shape))
            if node.attr == "T" and len(base.shape) == 2:
                return T((base.shape[1], base.shape[0]))
            if node.attr in ("data", "real"):
                return base
        if isinstance(base, U):
            return base
        return U("unsupported_op:." + node.attr)

    def ev_subscript(self, node: ast.Subscript, env: Dict[str, Any]) -> Any:
        base = self.ev(node.value, env)
        idx_node = node.slice
        if isinstance(idx_node, ast.Index):  # pragma: no cover - Python 3.8 only
            idx_node = idx_node.value  # type: ignore[attr-defined]
        if isinstance(base, Seq):
            idx = self.ev(idx_node, env)
            if isinstance(idx, I) and is_int(idx.value):
                try:
                    return base.items[idx.value]
                except IndexError:
                    return U("unknown")
            if isinstance(idx_node, ast.Slice):
                lo = self.ev(idx_node.lower, env).value if idx_node.lower is not None else None
                hi = self.ev(idx_node.upper, env).value if idx_node.upper is not None else None
                if (lo is None or is_int(lo)) and (hi is None or is_int(hi)) and idx_node.step is None:
                    return Seq(base.items[lo:hi])
            return U("unknown")
        if isinstance(base, Mod) and isinstance(base.ref, Layer) and base.ref.kind in ("Sequential", "ModuleList"):
            idx = self.ev(idx_node, env)
            if isinstance(idx, I) and is_int(idx.value):
                try:
                    return Mod(base.ref.children[idx.value])  # type: ignore[index]
                except IndexError:
                    return U("unknown")
        if isinstance(base, U):
            return base
        return U("unsupported_op:indexing")

    def _binop(self, a: Any, b: Any, op: ast.operator, node: ast.AST) -> Any:
        if isinstance(a, I) and isinstance(b, I):
            if is_int(a.value) and is_int(b.value):
                fn = {ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y, ast.Mult: lambda x, y: x * y,
                      ast.FloorDiv: lambda x, y: x // y if y else None, ast.Pow: lambda x, y: x ** y if 0 <= y <= 64 else None}.get(type(op))
                if fn is not None:
                    return I(fn(a.value, b.value))
            return I(None)
        if isinstance(op, ast.MatMult):
            return U("unsupported_op:@")
        if isinstance(a, U):
            return a
        if isinstance(b, U):
            return b
        if isinstance(a, T) and isinstance(b, T):
            return self._guard(node, lambda: broadcast(a.shape, b.shape))
        if isinstance(a, T) and isinstance(b, I):
            return a
        if isinstance(b, T) and isinstance(a, I):
            return b
        return U("unknown")

    def _guard(self, node: ast.AST, fn: Callable[[], Shape], layer: Optional[Layer] = None) -> Any:
        """Run a shape function; turn the result into a value and ShapeError into a diagnostic."""
        try:
            shape = fn()
        except ShapeError as e:
            self.report(node, e, layer)
            return U("after_error")
        return T(shape) if shape is not None else U("dynamic")

    def report(self, node: ast.AST, e: ShapeError, layer: Optional[Layer]) -> None:
        args = dict(e.info)
        fix = None
        if layer is not None:
            param = {"linear_in": "in_features", "conv_in": "in_channels", "bn_features": "num_features"}.get(e.code)
            if param and param in layer.arg_ranges:
                r = layer.arg_ranges[param]
                if param in layer.literal_args:
                    fix = {"line": r[0], "col": r[1], "endLine": r[2], "endCol": r[3], "newText": str(e.info["actual"])}
                d = {"line": r[0], "col": r[1], "endLine": r[2], "endCol": r[3], "code": e.code, "args": args}
            else:
                d = {"line": layer.line, "col": layer.col, "endLine": layer.end_line, "endCol": layer.end_col, "code": e.code, "args": args}
            if fix:
                d["fix"] = fix
            self.rec.diag(d)
        d2 = {"line": node.lineno - 1, "col": node.col_offset, "endLine": (node.end_lineno or node.lineno) - 1, "endCol": node.end_col_offset or 0, "code": e.code, "args": args}
        if fix:
            d2["fix"] = fix
        self.rec.diag(d2)

    # calls

    def ev_call(self, node: ast.Call, env: Dict[str, Any]) -> Any:
        func = node.func
        callee = None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self" and "self" not in env:
            callee = self.ev_attribute(func, env)
        elif isinstance(func, ast.Name) and func.id in env:
            callee = env[func.id]
        elif isinstance(func, ast.Subscript):
            callee = self.ev_subscript(func, env)
        if isinstance(callee, Mod):
            args = [self.ev(a, env) for a in node.args]
            kwargs = {k.arg: self.ev(k.value, env) for k in node.keywords if k.arg}
            return self.call_module(callee.ref, args, kwargs, node, self._mod_name(func))
        if isinstance(callee, Bound):
            fn = self.ctx.cls.methods[callee.name]
            args = [self.ev(a, env) for a in node.args]
            kwargs = {k.arg: self.ev(k.value, env) for k in node.keywords if k.arg}
            return self.an._call_function(fn, args, kwargs, self.ctx, self.rec, self.depth + 1, self.prefix)
        if isinstance(callee, U):
            return callee
        if isinstance(func, ast.Attribute):
            q = qualname(func, self.an.aliases)
            if q is not None and (q.startswith("torch.") or q == "torch"):
                return self.call_torch_function(q, node, env)
            base = self.ev(func.value, env)
            if isinstance(base, (T, U)):
                return self.call_method(base, func.attr, node, env)
            return U("unsupported_op:" + (dotted(func) or func.attr))
        if isinstance(func, ast.Name):
            q = qualname(func, self.an.aliases)
            if q and q.startswith("torch."):
                return self.call_torch_function(q, node, env)
            if func.id == "len":
                v = self.ev(node.args[0], env) if node.args else U("unknown")
                if isinstance(v, T) and v.shape:
                    return I(v.shape[0])
                if isinstance(v, Seq):
                    return I(len(v.items))
                return I(None)
            if func.id == "int" and node.args:
                v = self.ev(node.args[0], env)
                return v if isinstance(v, I) else I(None)
            return U("unsupported_op:" + func.id)
        return U("unsupported_op:call")

    def _mod_name(self, func: ast.AST) -> str:
        d = dotted(func)
        if d and d.startswith("self."):
            return self.prefix + d[5:]
        return self.prefix + (d or "?")

    def call_module(self, ref: Any, args: List[Any], kwargs: Dict[str, Any], node: ast.Call, name: str) -> Any:
        if isinstance(ref, UserRef):
            if ref.ctx is None:
                return U("unsupported_op:" + ref.cls.name)
            return self.an.run_model(ref.ctx, args, self.rec, self.depth + 1, name + ".")
        layer: Layer = ref
        x = args[0] if args else kwargs.get("input", U("no_input"))
        if layer.kind == "Sequential":
            out = x
            for cname, child in zip(layer.child_names or [], layer.children or []):
                out = self._apply_layer(child, out, node, name + "." + cname)
            self.rec.record(layer.end_line, (layer.line, layer.col), out)
            return out
        return self._apply_layer(layer, x, node, name)

    def _apply_layer(self, layer: Layer, x: Any, node: ast.Call, name: str) -> Any:
        if layer.kind == "Sequential":
            return self.call_module(layer, [x], {}, node, name)
        if isinstance(x, U):
            out: Any = x
            err = False
        elif not isinstance(x, T):
            out, err = U("unknown"), False
        else:
            try:
                shape = infer(layer, x.shape)
                if shape is None:
                    out = U(self._unknown_reason(layer))
                else:
                    out = T(shape)
                err = False
            except ShapeError as e:
                self.report(node, e, layer)
                out, err = U("after_error"), True
        self.rec.record(layer.end_line, (layer.line, layer.col), out)
        self.rec.block(name, layer, out, err)
        return out

    def _unknown_reason(self, layer: Layer) -> str:
        if layer.kind == "unknown":
            return "unsupported_op:" + layer.name
        if any(v is None for k, v in layer.params.items() if k not in ("stride", "padding_idx", "divisor_override")):
            return "ctor_arg"
        return "dynamic"

    # torch functions and tensor methods

    def _args(self, node: ast.Call, env: Dict[str, Any], names: Sequence[str], defaults: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        out = dict(defaults)
        for i, a in enumerate(node.args):
            if isinstance(a, ast.Starred):
                return None
            if i >= len(names):
                return None
            out[names[i]] = self.ev(a, env)
        for k in node.keywords:
            if k.arg is None:
                return None
            out[k.arg] = self.ev(k.value, env)
        return out

    def call_torch_function(self, q: str, node: ast.Call, env: Dict[str, Any]) -> Any:
        name = q.rsplit(".", 1)[-1]
        functional = q.startswith("torch.nn.functional.")
        if (functional or q.count(".") == 1) and name in _ELEMENTWISE:
            x = self.ev(node.args[0], env) if node.args else U("no_input")
            return x if isinstance(x, (T, U)) else U("unknown")
        if q.count(".") == 1 and name in _BINARY and len(node.args) >= 2:
            return self._binop(self.ev(node.args[0], env), self.ev(node.args[1], env), ast.Add(), node)
        if functional and name in _POOL_FN:
            nd, kind = _POOL_FN[name]
            a = self._args(node, env, ["input", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices"] if kind == "max" else ["input", "kernel_size", "stride", "padding", "ceil_mode"], {"stride": Seq(()), "padding": I(0), "dilation": I(1), "ceil_mode": I(False), "return_indices": I(False)})
            if a is None:
                return U("unsupported_op:" + name)
            x = a["input"]
            if isinstance(x, U):
                return x
            vals = {k: _to_py(v) for k, v in a.items() if k != "input"}
            if any(v is _BAD for v in vals.values()) or vals.get("return_indices"):
                return U("dynamic")
            stride = vals["stride"] if vals["stride"] not in ((), None) else None
            return self._guard(node, lambda: F.pool(x.shape, nd, kind, vals["kernel_size"], stride, vals["padding"], vals.get("dilation", 1), bool(vals["ceil_mode"])))
        if functional and name in ("adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_max_pool1d", "adaptive_max_pool2d"):
            a = self._args(node, env, ["input", "output_size", "return_indices"], {"return_indices": I(False)})
            if a is None or not isinstance(a.get("input"), T):
                return a["input"] if a and isinstance(a.get("input"), U) else U("dynamic")
            out = _to_py(a["output_size"])
            if out is _BAD or _to_py(a["return_indices"]):
                return U("dynamic")
            return self._guard(node, lambda: F.adaptive_pool(a["input"].shape, int(name[-2]), out))
        if not functional and q.count(".") == 1:
            if name in _TENSOR_LIKE_METHODS:
                if not node.args:
                    return U("no_input")
                x = self.ev(node.args[0], env)
                if isinstance(x, (T, U)):
                    fake = ast.Call(func=node.func, args=node.args[1:], keywords=node.keywords)
                    ast.copy_location(fake, node)
                    fake.end_lineno, fake.end_col_offset = node.end_lineno, node.end_col_offset  # type: ignore[attr-defined]
                    return self.call_method(x, name, fake, env, torch_fn=True)
            if name in ("cat", "concat", "concatenate", "stack"):
                a = self._args(node, env, ["tensors", "dim"], {"dim": I(0)})
                if a is None:
                    return U("unsupported_op:" + name)
                ts = a["tensors"]
                if not isinstance(ts, Seq):
                    return U("dynamic")
                for t in ts.items:
                    if isinstance(t, U):
                        return t
                    if not isinstance(t, T):
                        return U("unknown")
                dim = _to_py(a["dim"])
                if not is_int(dim):
                    return U("dynamic")
                shapes = [t.shape for t in ts.items]
                fn = F.stack if name == "stack" else F.cat
                return self._guard(node, lambda: fn(shapes, dim))
        return U("unsupported_op:" + ("F." if functional else "torch.") + name)

    def call_method(self, x: Any, m: str, node: ast.Call, env: Dict[str, Any], torch_fn: bool = False) -> Any:
        if m in ("size",):
            if isinstance(x, U):
                return I(None) if node.args else U(x.reason)
            a = self._args(node, env, ["dim"], {"dim": None})
            if a is None:
                return U("unknown")
            if a["dim"] is None:
                return Seq(tuple(I(d) for d in x.shape))
            d = _to_py(a["dim"])
            if is_int(d) and -len(x.shape) <= d < len(x.shape):
                return I(x.shape[d])
            return I(None)
        if m in ("dim", "ndimension"):
            return I(len(x.shape)) if isinstance(x, T) else I(None)
        if m == "numel":
            return I(numel(x.shape)) if isinstance(x, T) else I(None)
        if isinstance(x, U):
            if m in _SHAPE_KEEP or m in _VIEW_METHODS:
                return x
            return U("unsupported_op:." + m)
        if m in _SHAPE_KEEP:
            return x
        if m in ("view", "reshape"):
            if len(node.args) == 1 and not node.keywords:
                v = self.ev(node.args[0], env)
                target = _to_py(v) if isinstance(v, Seq) else (_to_py(v),)
            else:
                target = tuple(_to_py(self.ev(a, env)) for a in node.args)
                for k in node.keywords:
                    if k.arg == "shape":
                        target = _to_py(self.ev(k.value, env))
            if target is _BAD or not isinstance(target, tuple) or any(t is _BAD for t in target):
                return U("dynamic")
            if any(not (is_int(t) or isinstance(t, str)) for t in target):
                return U("dynamic")
            return self._guard(node, lambda: F.reshape(x.shape, target))
        if m == "view_as":
            other = self.ev(node.args[0], env) if node.args else U("unknown")
            if isinstance(other, T):
                return self._guard(node, lambda: F.reshape(x.shape, other.shape))
            return U("dynamic")
        if m == "flatten":
            a = self._args(node, env, ["start_dim", "end_dim"], {"start_dim": I(0), "end_dim": I(-1)})
            if a is None:
                return U("dynamic")
            sd, ed = _to_py(a["start_dim"]), _to_py(a["end_dim"])
            if not (is_int(sd) and is_int(ed)):
                return U("dynamic")
            return self._guard(node, lambda: F.flatten(x.shape, sd, ed))
        if m == "squeeze":
            a = self._args(node, env, ["dim"], {"dim": None})
            if a is None:
                return U("dynamic")
            d = None if a["dim"] is None else _to_py(a["dim"])
            if d is _BAD:
                return U("dynamic")
            return self._guard(node, lambda: F.squeeze(x.shape, d))
        if m == "unsqueeze":
            a = self._args(node, env, ["dim"], {})
            d = _to_py(a["dim"]) if a and "dim" in a else _BAD
            if not is_int(d):
                return U("dynamic")
            return self._guard(node, lambda: F.unsqueeze(x.shape, d))
        if m == "permute":
            if len(node.args) == 1:
                dims = _to_py(self.ev(node.args[0], env))
            else:
                dims = tuple(_to_py(self.ev(a, env)) for a in node.args)
            for k in node.keywords:
                if k.arg == "dims":
                    dims = _to_py(self.ev(k.value, env))
            if not isinstance(dims, tuple) or not all(is_int(d) for d in dims):
                return U("dynamic")
            return self._guard(node, lambda: F.permute(x.shape, dims))
        if m in ("transpose", "swapaxes", "swapdims"):
            a = self._args(node, env, ["dim0", "dim1"], {})
            if a is None or "dim0" not in a or "dim1" not in a:
                return U("dynamic")
            d0, d1 = _to_py(a["dim0"]), _to_py(a["dim1"])
            if not (is_int(d0) and is_int(d1)):
                return U("dynamic")
            return self._guard(node, lambda: F.transpose(x.shape, d0, d1))
        if m == "t":
            if len(x.shape) <= 1:
                return x
            if len(x.shape) == 2:
                return T((x.shape[1], x.shape[0]))
            return U("dynamic")
        if m in ("mean", "sum", "amax", "amin", "prod", "std", "var", "logsumexp", "norm"):
            a = self._args(node, env, ["dim", "keepdim"], {"dim": None, "keepdim": I(False)})
            if a is None:
                return U("dynamic")
            d = None if a["dim"] is None else _to_py(a["dim"])
            kd = _to_py(a["keepdim"])
            if d is _BAD or kd is _BAD or (m in ("prod",) and isinstance(d, tuple)):
                return U("dynamic")
            if m == "logsumexp" and d is None:
                return U("dynamic")
            return self._guard(node, lambda: F.reduce(x.shape, d, bool(kd)))
        if m in ("softmax", "log_softmax"):
            return x
        if m in _BINARY and node.args:
            return self._binop(x, self.ev(node.args[0], env), ast.Add(), node)
        return U("unsupported_op:." + m)


# ---- helpers ---------------------------------------------------------------------------------

_BAD = object()

_ELEMENTWISE = frozenset(
    """relu relu_ relu6 leaky_relu gelu elu selu celu silu mish sigmoid tanh hardtanh hardswish
    hardsigmoid softplus softsign logsigmoid softmax log_softmax softmin dropout dropout1d dropout2d
    dropout3d alpha_dropout feature_alpha_dropout normalize layer_norm batch_norm instance_norm group_norm
    exp log log2 log10 log1p abs sqrt rsqrt square neg clamp clip sin cos sign round floor ceil
    sigmoid_ tanh_ erf reciprocal tanhshrink softshrink hardshrink threshold rrelu""".split()
)
_BINARY = frozenset("add sub mul div true_divide maximum minimum pow fmod remainder".split())
_POOL_FN = {
    "max_pool1d": (1, "max"),
    "max_pool2d": (2, "max"),
    "avg_pool1d": (1, "avg"),
    "avg_pool2d": (2, "avg"),
}
_SHAPE_KEEP = frozenset(
    """contiguous float double half bfloat16 long int bool to cuda cpu detach clone relu relu_ sigmoid
    tanh exp log abs sqrt neg clamp clamp_ clip requires_grad_ type_as sigmoid_ tanh_ square float_
    div_ mul_ add_ sub_ round floor ceil""".split()
)
_VIEW_METHODS = frozenset("view reshape flatten squeeze unsqueeze permute transpose t mean sum".split())
_TENSOR_LIKE_METHODS = frozenset("flatten reshape squeeze unsqueeze permute transpose swapaxes mean sum amax amin".split())


def _to_py(v: Any) -> Any:
    if isinstance(v, I):
        return v.value if v.value is not None else _BAD
    if isinstance(v, Seq):
        items = tuple(_to_py(x) for x in v.items)
        return _BAD if any(x is _BAD for x in items) else items
    return _BAD


def _py_to_value(v: Any) -> Any:
    if isinstance(v, tuple):
        return Seq(tuple(_py_to_value(x) for x in v))
    if v is None:
        return U("unknown")
    return I(v)


def _const_bool(test: ast.AST, consts: Dict[str, Any], self_consts: Dict[str, Any], env: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    """Statically decide an ``if`` condition when it only uses known constants."""
    node = test
    negate = False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        node, negate = node.operand, True
    val: Any = None
    if isinstance(node, ast.Constant):
        val = node.value
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        if node.attr in self_consts and isinstance(self_consts[node.attr], (bool, int)):
            val = self_consts[node.attr]
        else:
            return None
    elif isinstance(node, ast.Name) and node.id in consts and isinstance(consts[node.id], (bool, int)):
        val = consts[node.id]
    else:
        return None
    if not isinstance(val, (bool, int)) or val is None:
        return None
    return (not val) if negate else bool(val)


def _module_consts(tree: ast.Module) -> Dict[str, Any]:
    consts: Dict[str, Any] = {}
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            v = eval_const(st.value, consts)
            consts[st.targets[0].id] = v
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            consts[st.target.id] = eval_const(st.value, consts)
    return consts


def _bind_ctor(init: ast.FunctionDef, args: Sequence[ast.AST], kwargs: Dict[str, ast.AST], caller_consts: Dict[str, Any], module_consts: Dict[str, Any]) -> Dict[str, Any]:
    params = init.args.args[1:]
    defaults = init.args.defaults
    first_default = len(params) - len(defaults)
    bound: Dict[str, Any] = {}
    for i, p in enumerate(params):
        if i < len(args):
            bound[p.arg] = eval_const(args[i], caller_consts)
        elif p.arg in kwargs:
            bound[p.arg] = eval_const(kwargs[p.arg], caller_consts)
        elif i >= first_default:
            bound[p.arg] = eval_const(defaults[i - first_default], module_consts)
        else:
            bound[p.arg] = None
    for kw, d in zip(init.args.kwonlyargs, init.args.kw_defaults):
        if kw.arg in kwargs:
            bound[kw.arg] = eval_const(kwargs[kw.arg], caller_consts)
        elif d is not None:
            bound[kw.arg] = eval_const(d, module_consts)
        else:
            bound[kw.arg] = None
    return bound


def _parameter_numel(call: ast.Call, consts: Dict[str, Any], aliases: Dict[str, str]) -> Any:
    """``nn.Parameter(torch.zeros(3, 4))`` -> 12; unknown -> None; not a Parameter -> "no"."""
    q = qualname(call.func, aliases) or ""
    if not (q.endswith("nn.Parameter") or q.endswith("nn.parameter.Parameter")):
        return "no"
    if not call.args or not isinstance(call.args[0], ast.Call):
        return None
    inner = call.args[0]
    iq = qualname(inner.func, aliases) or ""
    if iq.rsplit(".", 1)[-1] not in ("zeros", "ones", "randn", "rand", "empty", "full") or not iq.startswith("torch."):
        return None
    dims_nodes = inner.args
    if iq.endswith("full"):
        dims_nodes = inner.args[:1]
    if len(dims_nodes) == 1:
        v = eval_const(dims_nodes[0], consts)
        dims = v if isinstance(v, tuple) else (v,)
    else:
        dims = tuple(eval_const(a, consts) for a in dims_nodes)
    total = 1
    for d in dims:
        if not is_int(d):
            return None
        total *= d
    return total


def _leaves(layer: Layer, name: str) -> List[Tuple[str, Layer]]:
    if layer.kind in ("Sequential", "ModuleList"):
        out: List[Tuple[str, Layer]] = []
        for cname, child in zip(layer.child_names or [], layer.children or []):
            out.extend(_leaves(child, name + "." + cname))
        return out
    return [(name, layer)]


def _memory(total_params: Optional[int], outputs: List[Shape]) -> Optional[Dict[str, int]]:
    if total_params is None or not outputs:
        return None
    acts = 0
    for s in outputs:
        n = numel(s)
        if n is None:
            return None
        acts += n
    p = total_params * 4
    mem = {"params": p, "grads": p, "optimizer": 2 * p, "activations": acts * 4}
    mem["total"] = sum(mem.values())
    return mem


def analyze_torch(tree: ast.Module, text: str) -> List[Dict[str, Any]]:
    return Analyzer(tree, text).analyze()
