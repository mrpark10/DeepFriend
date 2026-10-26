"""Formula results must equal real PyTorch on CPU, including when it raises."""
import os
import random

import pytest

from deepfriend.shapes.dims import ShapeError
from deepfriend.shapes import torch_formulas as tf

torch = pytest.importorskip("torch")
nn = torch.nn

N = int(os.environ.get("DF_PARITY_N", "400"))


def real(build, shape):
    """Run real torch; return output shape or 'raise'."""
    try:
        mod = build()
        if isinstance(mod, nn.Module):
            mod.eval()
        with torch.no_grad():
            out = mod(torch.zeros(*shape)) if callable(mod) else mod
        return tuple(out.shape)
    except Exception:
        return "raise"


def ours(fn):
    try:
        return fn()
    except ShapeError:
        return "raise"


def check(cases):
    bad = [c for c in cases if c[1] != c[2]]
    assert not bad, "%d mismatches, first: %r" % (len(bad), bad[:3])


def _tup(r, nd, lo, hi):
    if r.random() < 0.5:
        return r.randint(lo, hi)
    return tuple(r.randint(lo, hi) for _ in range(nd))


def test_conv_parity():
    r = random.Random(0)
    cases = []
    for _ in range(N):
        nd = r.choice([1, 2])
        groups = r.choice([1, 1, 1, 2, 3])
        cin = groups * r.randint(1, 3) + (r.random() < 0.05)  # occasionally invalid
        cout = groups * r.randint(1, 3) + (r.random() < 0.05)
        k = _tup(r, nd, 1, 5)
        s = _tup(r, nd, 1, 3)
        d = _tup(r, nd, 1, 2)
        pad = r.choice([0, 1, 2, "same", "valid", None])
        if pad is None:
            pad = _tup(r, nd, 0, 2)
        batched = r.random() < 0.8
        actual_c = cin if r.random() < 0.9 else cin + 1
        spatial = tuple(r.randint(1, 20) for _ in range(nd))
        shape = ((r.randint(1, 3),) if batched else ()) + (actual_c,) + spatial
        Conv = nn.Conv1d if nd == 1 else nn.Conv2d
        build = lambda: Conv(cin, cout, k, stride=s, padding=pad, dilation=d, groups=groups)
        mine = ours(lambda: tf.conv(shape, nd=nd, in_channels=cin, out_channels=cout, kernel_size=k, stride=s, padding=pad, dilation=d, groups=groups))
        cases.append(((nd, cin, cout, k, s, pad, d, groups, shape), mine, real(build, shape)))
    check(cases)


def test_pool_parity():
    r = random.Random(1)
    cases = []
    for _ in range(N):
        nd = r.choice([1, 2])
        kind = r.choice(["max", "avg"])
        k = _tup(r, nd, 1, 5)
        s = r.choice([None, _tup(r, nd, 1, 3)])
        p = _tup(r, nd, 0, 2)
        d = _tup(r, nd, 1, 2) if kind == "max" else 1
        ceil = r.random() < 0.5
        batched = r.random() < 0.8
        shape = ((r.randint(1, 3),) if batched else ()) + (r.randint(1, 4),) + tuple(r.randint(1, 20) for _ in range(nd))
        if kind == "max":
            M = nn.MaxPool1d if nd == 1 else nn.MaxPool2d
            build = lambda: M(k, stride=s, padding=p, dilation=d, ceil_mode=ceil)
        else:
            A = nn.AvgPool1d if nd == 1 else nn.AvgPool2d
            build = lambda: A(k, stride=s, padding=p, ceil_mode=ceil)
        mine = ours(lambda: tf.pool(shape, nd=nd, kind=kind, kernel_size=k, stride=s, padding=p, dilation=d, ceil_mode=ceil))
        cases.append(((nd, kind, k, s, p, d, ceil, shape), mine, real(build, shape)))
    check(cases)


def test_adaptive_parity():
    r = random.Random(2)
    cases = []
    for _ in range(N // 2):
        nd = r.choice([1, 2])
        if nd == 1:
            out = r.randint(1, 6)
        else:
            out = r.choice([r.randint(1, 6), (r.choice([None, r.randint(1, 6)]), r.randint(1, 6))])
        shape = (r.randint(1, 3), r.randint(1, 4)) + tuple(r.randint(1, 12) for _ in range(nd))
        kind = r.choice(["avg", "max"])
        if kind == "avg":
            M = nn.AdaptiveAvgPool1d if nd == 1 else nn.AdaptiveAvgPool2d
        else:
            M = nn.AdaptiveMaxPool1d if nd == 1 else nn.AdaptiveMaxPool2d
        mine = ours(lambda: tf.adaptive_pool(shape, nd=nd, output_size=out))
        cases.append(((nd, out, shape), mine, real(lambda: M(out), shape)))
    check(cases)


def test_flatten_reshape_parity():
    r = random.Random(3)
    cases = []
    for _ in range(N):
        rank = r.randint(1, 4)
        shape = tuple(r.randint(1, 5) for _ in range(rank))
        if r.random() < 0.5:
            sd = r.randint(-rank, rank - 1)
            ed = r.randint(-rank, rank - 1)
            mine = ours(lambda: tf.flatten(shape, start_dim=sd, end_dim=ed))
            cases.append((("flatten", shape, sd, ed), mine, real(lambda: nn.Flatten(sd, ed), shape)))
        else:
            total = 1
            for v in shape:
                total *= v
            tgt = [r.randint(1, 6) for _ in range(r.randint(1, 3))]
            if r.random() < 0.6:
                tgt[r.randrange(len(tgt))] = -1
            tgt = tuple(tgt)
            mine = ours(lambda: tf.reshape(shape, tgt))
            cases.append((("reshape", shape, tgt), mine, real(lambda: (lambda x: x.reshape(*tgt)), shape)))
    check(cases)


def test_cat_parity():
    r = random.Random(4)
    cases = []
    for _ in range(N // 2):
        rank = r.randint(1, 3)
        a = tuple(r.randint(1, 4) for _ in range(rank))
        b = list(a)
        dim = r.randint(-rank, rank - 1)
        b[dim] = r.randint(1, 4)
        if r.random() < 0.3:
            j = r.randrange(rank)
            b[j] = b[j] + 1
        b = tuple(b)
        mine = ours(lambda: tf.cat([a, b], dim=dim))
        def build():
            return lambda x: torch.cat([torch.zeros(*a), torch.zeros(*b)], dim=dim)
        cases.append(((a, b, dim), mine, real(build, (1,))))
    check(cases)


def test_linear_norm_parity():
    r = random.Random(5)
    cases = []
    for _ in range(N // 2):
        rank = r.randint(1, 3)
        shape = tuple(r.randint(1, 6) for _ in range(rank))
        fin = shape[-1] if r.random() < 0.8 else shape[-1] + 1
        mine = ours(lambda: tf.linear(shape, in_features=fin, out_features=3))
        cases.append((("linear", shape, fin), mine, real(lambda: nn.Linear(fin, 3), shape)))
        bshape = (r.randint(2, 3),) + tuple(r.randint(1, 5) for _ in range(r.randint(1, 3)))
        nd = r.choice([1, 2])
        nf = bshape[1] if r.random() < 0.8 else bshape[1] + 1
        B = nn.BatchNorm1d if nd == 1 else nn.BatchNorm2d
        mine = ours(lambda: tf.batchnorm(bshape, nd=nd, num_features=nf))
        cases.append((("bn", nd, bshape, nf), mine, real(lambda: B(nf), bshape)))
    check(cases)
