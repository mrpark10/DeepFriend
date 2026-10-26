"""PyTorch output-shape formulas.

Every function takes the input shape first and returns the output shape,
``None`` when it cannot be known statically, or raises ``ShapeError`` exactly
when PyTorch itself would raise. The rules follow the PyTorch source
(``aten/src/ATen/native/Pool.h``, ``ConvUtils.h``) and are checked against
real PyTorch by ``tests/shapes/test_torch_parity.py``.
"""

from typing import Any, List, Optional, Sequence, Tuple

from .dims import Shape, ShapeError, as_tuple, factorize, fmt_shape, is_int, norm_dim


def _spatial(in_shape: Tuple, nd: int, layer: str) -> Tuple[Tuple, Any, Tuple]:
    """Split (N?, C, *spatial) and validate the rank (batched or unbatched)."""
    r = len(in_shape)
    if r not in (nd + 1, nd + 2):
        raise ShapeError("rank", layer=layer, expected="%dD/%dD" % (nd + 1, nd + 2), actual=r)
    lead = in_shape[: r - nd - 1]
    return lead, in_shape[r - nd - 1], in_shape[r - nd :]


def _check_positive(name: str, values: Sequence[Any], layer: str, allow_zero: bool = False) -> None:
    for v in values:
        if not is_int(v) or v < (0 if allow_zero else 1):
            raise ShapeError("invalid_arg", layer=layer, reason=name, value=v)


# ---- linear / conv ------------------------------------------------------------


def linear(in_shape: Shape, in_features: int, out_features: int) -> Shape:
    if in_shape is None:
        return None
    if len(in_shape) == 0:
        raise ShapeError("rank", layer="Linear", expected="1D+", actual=0)
    last = in_shape[-1]
    if is_int(last) and is_int(in_features) and last != in_features:
        raise ShapeError("linear_in", expected=in_features, actual=last)
    return tuple(in_shape[:-1]) + (out_features,)


def conv(
    in_shape: Shape,
    nd: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Shape:
    layer = "Conv%dd" % nd
    k = as_tuple(kernel_size, nd)
    s = as_tuple(stride, nd)
    d = as_tuple(dilation, nd)
    _check_positive("kernel_size", k, layer)
    _check_positive("stride", s, layer)
    _check_positive("dilation", d, layer)
    if not is_int(groups) or groups < 1 or in_channels % groups or out_channels % groups:
        raise ShapeError("invalid_arg", layer=layer, reason="groups", value=groups)
    if isinstance(padding, str):
        if padding not in ("same", "valid"):
            raise ShapeError("invalid_arg", layer=layer, reason="padding", value=padding)
        if padding == "same" and any(x != 1 for x in s):
            raise ShapeError("invalid_arg", layer=layer, reason="same_stride", value=stride)
        same = padding == "same"
        p = (0,) * nd
    else:
        same = False
        p = as_tuple(padding, nd)
        _check_positive("padding", p, layer, allow_zero=True)
    if in_shape is None:
        return None
    lead, c, spatial = _spatial(in_shape, nd, layer)
    if is_int(c) and c != in_channels:
        raise ShapeError("conv_in", expected=in_channels, actual=c)
    out: List[Any] = []
    for i, L in enumerate(spatial):
        if not is_int(L):
            return None
        if same:
            out.append(L)
            continue
        eff = d[i] * (k[i] - 1) + 1
        if L + 2 * p[i] < eff:
            raise ShapeError("nonpositive", layer=layer)
        out.append((L + 2 * p[i] - eff) // s[i] + 1)
    return tuple(lead) + (out_channels,) + tuple(out)


# ---- pooling ------------------------------------------------------------------


def _pool_out(L: int, k: int, s: int, p: int, d: int, ceil_mode: bool) -> int:
    num = L + 2 * p - d * (k - 1) - 1 + ((s - 1) if ceil_mode else 0)
    out = num // s + 1
    if ceil_mode and (out - 1) * s >= L + p:
        out -= 1
    return out


def pool(
    in_shape: Shape,
    nd: int,
    kind: str,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = 0,
    dilation: Any = 1,
    ceil_mode: bool = False,
) -> Shape:
    layer = ("MaxPool%dd" if kind == "max" else "AvgPool%dd") % nd
    k = as_tuple(kernel_size, nd)
    s = k if stride is None or (isinstance(stride, (tuple, list)) and len(stride) == 0) else as_tuple(stride, nd)
    p = as_tuple(padding, nd)
    d = as_tuple(dilation, nd) if kind == "max" else (1,) * nd
    _check_positive("kernel_size", k, layer)
    _check_positive("stride", s, layer)
    _check_positive("dilation", d, layer)
    _check_positive("padding", p, layer, allow_zero=True)
    for i in range(nd):
        limit = k[i] // 2  # PyTorch: "pad should be at most half of kernel size" (dilation ignored)
        if p[i] > limit:
            raise ShapeError("invalid_arg", layer=layer, reason="pool_padding", value=p[i])
    if in_shape is None:
        return None
    lead, c, spatial = _spatial(in_shape, nd, layer)
    out: List[Any] = []
    for i, L in enumerate(spatial):
        if not is_int(L):
            return None
        o = _pool_out(L, k[i], s[i], p[i], d[i], ceil_mode)
        if o < 1:
            raise ShapeError("nonpositive", layer=layer)
        out.append(o)
    return tuple(lead) + (c,) + tuple(out)


def adaptive_pool(in_shape: Shape, nd: int, output_size: Any) -> Shape:
    layer = "AdaptivePool%dd" % nd
    o = as_tuple(output_size, nd)
    if in_shape is None:
        return None
    lead, c, spatial = _spatial(in_shape, nd, layer)
    out = tuple(spatial[i] if o[i] is None else o[i] for i in range(nd))
    return tuple(lead) + (c,) + out


# ---- normalization / embedding -------------------------------------------------


def batchnorm(in_shape: Shape, nd: int, num_features: int) -> Shape:
    if in_shape is None:
        return None
    allowed = (2, 3) if nd == 1 else (nd + 2,)
    if len(in_shape) not in allowed:
        raise ShapeError("rank", layer="BatchNorm%dd" % nd, expected="/".join("%dD" % a for a in allowed), actual=len(in_shape))
    c = in_shape[1]
    if is_int(c) and c != num_features:
        raise ShapeError("bn_features", expected=num_features, actual=c)
    return tuple(in_shape)


def layernorm(in_shape: Shape, normalized_shape: Any) -> Shape:
    if in_shape is None:
        return None
    ns = tuple(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else (normalized_shape,)
    tail = tuple(in_shape[len(in_shape) - len(ns) :]) if len(ns) <= len(in_shape) else None
    if tail is None:
        raise ShapeError("invalid_arg", layer="LayerNorm", reason="normalized_shape", value=fmt_shape(ns))
    for a, b in zip(tail, ns):
        if is_int(a) and is_int(b) and a != b:
            raise ShapeError("invalid_arg", layer="LayerNorm", reason="normalized_shape", value=fmt_shape(ns))
    return tuple(in_shape)


def embedding(in_shape: Shape, num_embeddings: int, embedding_dim: int) -> Shape:
    if in_shape is None:
        return None
    return tuple(in_shape) + (embedding_dim,)


def identity(in_shape: Shape) -> Shape:
    return in_shape


# ---- view-like ops -------------------------------------------------------------


def flatten(in_shape: Shape, start_dim: int = 1, end_dim: int = -1) -> Shape:
    if in_shape is None:
        return None
    r = len(in_shape)
    if r == 0:
        norm_dim(start_dim, 0)
        norm_dim(end_dim, 0)
        return (1,)
    sd = norm_dim(start_dim, r)
    ed = norm_dim(end_dim, r)
    if sd > ed:
        raise ShapeError("invalid_arg", layer="Flatten", reason="start_after_end", value="%s>%s" % (start_dim, end_dim))
    if sd == ed:
        return tuple(in_shape)
    coef, syms = factorize(in_shape[sd : ed + 1])
    if syms:
        if sum(syms.values()) == 1 and coef == 1:
            mid: Any = next(iter(syms))
        else:
            return None
    else:
        mid = coef
    return tuple(in_shape[:sd]) + (mid,) + tuple(in_shape[ed + 1 :])


def reshape(in_shape: Shape, target: Sequence[Any]) -> Shape:
    if in_shape is None:
        return None
    target = tuple(target)
    to_str = fmt_shape(target)
    if any(t is None for t in target):
        return None
    if sum(1 for t in target if t == -1) > 1:
        raise ShapeError("invalid_arg", layer="view", reason="multiple_minus_one", value=to_str)
    if any(is_int(t) and t < -1 for t in target):
        raise ShapeError("invalid_arg", layer="view", reason="negative", value=to_str)
    in_coef, in_syms = factorize(in_shape)
    known = [t for t in target if t != -1]
    t_coef, t_syms = factorize(known)
    frm: Any = in_coef if not in_syms else fmt_shape(in_shape)
    if -1 not in target:
        if in_syms != t_syms:
            return None
        if in_coef != t_coef:
            raise ShapeError("reshape_numel", **{"from": frm, "to": to_str})
        return target
    # infer -1
    rest_syms = in_syms - t_syms
    if t_syms - in_syms:
        return None
    if t_coef == 0 or in_coef % t_coef:
        raise ShapeError("reshape_numel", **{"from": frm, "to": to_str})
    ratio = in_coef // t_coef
    if not rest_syms:
        fill: Any = ratio
    elif sum(rest_syms.values()) == 1 and ratio == 1:
        fill = next(iter(rest_syms))
    else:
        return None
    return tuple(fill if t == -1 else t for t in target)


def cat(shapes: Sequence[Shape], dim: int = 0) -> Shape:
    if any(s is None for s in shapes) or not shapes:
        return None
    r = len(shapes[0])  # type: ignore[arg-type]
    if any(len(s) != r for s in shapes):  # type: ignore[arg-type]
        raise ShapeError("cat_mismatch", dim="rank", a=fmt_shape(shapes[0]), b=fmt_shape(next(s for s in shapes if len(s) != r)))  # type: ignore[arg-type]
    dd = norm_dim(dim, r)
    for j in range(r):
        if j == dd:
            continue
        vals = [s[j] for s in shapes]  # type: ignore[index]
        ints = [v for v in vals if is_int(v)]
        if len(set(ints)) > 1:
            other = next(s for s in shapes if is_int(s[j]) and s[j] != ints[0])  # type: ignore[index]
            raise ShapeError("cat_mismatch", dim=j, a=fmt_shape(shapes[0]), b=fmt_shape(other))
        if len(set(map(str, vals))) > 1:
            return None
    total = 0
    for s in shapes:
        v = s[dd]  # type: ignore[index]
        if not is_int(v):
            return None
        total += v
    out = list(shapes[0])  # type: ignore[arg-type]
    out[dd] = total
    return tuple(out)


def stack(shapes: Sequence[Shape], dim: int = 0) -> Shape:
    if any(s is None for s in shapes) or not shapes:
        return None
    first = shapes[0]
    for s in shapes[1:]:
        if len(s) != len(first) or any(is_int(a) and is_int(b) and a != b for a, b in zip(s, first)):  # type: ignore[arg-type]
            raise ShapeError("cat_mismatch", dim="all", a=fmt_shape(first), b=fmt_shape(s))
    r = len(first) + 1  # type: ignore[arg-type]
    dd = norm_dim(dim, r)
    out = list(first)  # type: ignore[arg-type]
    out.insert(dd, len(shapes))
    return tuple(out)


def squeeze(in_shape: Shape, dim: Any = None) -> Shape:
    if in_shape is None:
        return None
    r = len(in_shape)
    if dim is None:
        if any(not is_int(d) for d in in_shape):
            return None
        return tuple(d for d in in_shape if d != 1)
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    idx = {norm_dim(x, r) for x in dims}
    out = []
    for i, d in enumerate(in_shape):
        if i in idx and not is_int(d):
            return None
        if i in idx and d == 1:
            continue
        out.append(d)
    return tuple(out)


def unsqueeze(in_shape: Shape, dim: int) -> Shape:
    if in_shape is None:
        return None
    r = len(in_shape) + 1
    dd = norm_dim(dim, r)
    out = list(in_shape)
    out.insert(dd, 1)
    return tuple(out)


def permute(in_shape: Shape, dims: Sequence[int]) -> Shape:
    if in_shape is None:
        return None
    r = len(in_shape)
    if len(dims) != r:
        raise ShapeError("invalid_arg", layer="permute", reason="dims_len", value=len(dims))
    nd = [norm_dim(x, r) for x in dims]
    if len(set(nd)) != r:
        raise ShapeError("invalid_arg", layer="permute", reason="dims_repeat", value=str(tuple(dims)))
    return tuple(in_shape[i] for i in nd)


def transpose(in_shape: Shape, dim0: int, dim1: int) -> Shape:
    if in_shape is None:
        return None
    r = len(in_shape)
    a, b = norm_dim(dim0, r), norm_dim(dim1, r)
    out = list(in_shape)
    out[a], out[b] = out[b], out[a]
    return tuple(out)


def reduce(in_shape: Shape, dim: Any = None, keepdim: bool = False) -> Shape:
    if in_shape is None:
        return None
    r = len(in_shape)
    if dim is None:
        return (1,) * r if keepdim else ()
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    idx = {norm_dim(x, r) for x in dims}
    out: List[Any] = []
    for i, d in enumerate(in_shape):
        if i in idx:
            if keepdim:
                out.append(1)
        else:
            out.append(d)
    return tuple(out)


def optional_int(v: Any) -> Optional[int]:
    return v if is_int(v) else None
