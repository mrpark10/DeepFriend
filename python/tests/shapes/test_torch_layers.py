import ast
import random

import pytest

from deepfriend.shapes.torch_layers import (
    ACTIVATIONS,
    collect_aliases,
    eval_const,
    infer,
    param_count,
    parse_layer,
)


def first_call(src, consts=None, aliases=None):
    tree = ast.parse(src)
    al = aliases if aliases is not None else collect_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            return parse_layer(node, consts or {}, al)
    raise AssertionError("no call")


HDR = "import torch.nn as nn\n"


def test_positional_equals_keyword():
    a = first_call(HDR + "nn.Conv2d(3, 64, 7, 2, 3)")
    b = first_call(HDR + "nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)")
    assert a.kind == b.kind == "Conv2d"
    assert a.params == b.params
    assert a.params["dilation"] == 1 and a.params["groups"] == 1 and a.params["bias"] is True


def test_arg_range_points_at_literal():
    src = HDR + "fc = nn.Linear(10000, 10)"
    layer = first_call(src)
    line, col, eline, ecol = layer.arg_ranges["in_features"]
    text = src.splitlines()[line].encode("utf-8")
    assert text[col:ecol].decode() == "10000"
    assert (line, eline) == (1, 1)


def test_aliases_variants():
    assert first_call("from torch import nn\nnn.ReLU()").kind == "ReLU"
    assert first_call("import torch\ntorch.nn.Linear(3, 4)").kind == "Linear"
    assert first_call("from torch.nn import Linear as L\nL(3, 4)").kind == "Linear"
    assert first_call("import numpy as np\nnp.zeros(3)") is None


def test_const_expressions():
    layer = first_call(HDR + "nn.Linear(64 * 7 * 7, hidden)", consts={"hidden": 128})
    assert layer.params["in_features"] == 3136 and layer.params["out_features"] == 128


def test_unresolved_param_is_none():
    layer = first_call(HDR + "nn.Linear(n, 10)")
    assert layer.params["in_features"] is None


def test_eval_const():
    expr = lambda s, c=None: eval_const(ast.parse(s, mode="eval").body, c or {})
    assert expr("(3, 3)") == (3, 3)
    assert expr("-1") == -1
    assert expr("2 ** 3") == 8
    assert expr("a // 2", {"a": 9}) == 4
    assert expr("'same'") == "same"
    assert expr("foo()") is None


def test_sequential_children():
    src = HDR + "s = nn.Sequential(\n    nn.Conv2d(1, 8, 3),\n    nn.ReLU(),\n    nn.Flatten(),\n)"
    layer = first_call(src)
    assert layer.kind == "Sequential"
    assert [c.kind for c in layer.children] == ["Conv2d", "ReLU", "Flatten"]
    assert [c.line for c in layer.children] == [2, 3, 4]


def test_modulelist_children():
    layer = first_call(HDR + "nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])")
    assert layer.kind == "ModuleList" and len(layer.children) == 2


def test_unknown_nn_layer():
    layer = first_call(HDR + "nn.TransformerEncoderLayer(512, 8)")
    assert layer.kind == "unknown" and layer.name == "TransformerEncoderLayer"
    assert infer(layer, (2, 10, 512)) is None
    assert param_count(layer) is None


def test_infer_dispatch():
    assert infer(first_call(HDR + "nn.Conv2d(3, 16, 3, padding=1)"), (2, 3, 8, 8)) == (2, 16, 8, 8)
    assert infer(first_call(HDR + "nn.MaxPool2d(2)"), (2, 16, 8, 8)) == (2, 16, 4, 4)
    assert infer(first_call(HDR + "nn.Flatten()"), (2, 16, 4, 4)) == (2, 256)
    assert infer(first_call(HDR + "nn.ReLU(inplace=True)"), (2, 256)) == (2, 256)
    assert infer(first_call(HDR + "nn.MaxPool2d(2, return_indices=True)"), (2, 16, 8, 8)) is None
    seq = first_call(HDR + "nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Flatten())")
    assert infer(seq, (2, 1, 5, 5)) == (2, 36)


def test_activations_set():
    for n in ["ReLU", "GELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "Dropout", "Dropout2d", "Identity", "LeakyReLU", "SiLU"]:
        assert n in ACTIVATIONS


@pytest.mark.parametrize(
    "src,expected",
    [
        ("nn.Linear(10, 5)", 55),
        ("nn.Linear(10, 5, bias=False)", 50),
        ("nn.Conv2d(3, 16, 3)", 3 * 16 * 9 + 16),
        ("nn.Conv2d(8, 16, (3, 5), groups=4, bias=False)", 16 * 2 * 15),
        ("nn.Conv1d(4, 6, 5)", 4 * 6 * 5 + 6),
        ("nn.BatchNorm2d(32)", 64),
        ("nn.BatchNorm1d(32, affine=False)", 0),
        ("nn.LayerNorm(16)", 32),
        ("nn.LayerNorm((4, 8), elementwise_affine=False)", 0),
        ("nn.Embedding(100, 8)", 800),
        ("nn.PReLU()", 1),
        ("nn.ReLU()", 0),
        ("nn.MaxPool2d(2)", 0),
        ("nn.Flatten()", 0),
        ("nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))", 15 + 8),
    ],
)
def test_param_count(src, expected):
    assert param_count(first_call(HDR + src)) == expected


def test_param_count_parity():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    r = random.Random(7)
    for _ in range(200):
        kind = r.choice(["Linear", "Conv2d", "Conv1d", "BatchNorm2d", "LayerNorm", "Embedding"])
        if kind == "Linear":
            a = (r.randint(1, 50), r.randint(1, 50))
            kw = {"bias": r.random() < 0.5}
        elif kind in ("Conv2d", "Conv1d"):
            g = r.choice([1, 2])
            a = (g * r.randint(1, 4), g * r.randint(1, 4), r.randint(1, 5))
            kw = {"groups": g, "bias": r.random() < 0.5}
        elif kind == "BatchNorm2d":
            a = (r.randint(1, 64),)
            kw = {"affine": r.random() < 0.5}
        elif kind == "LayerNorm":
            a = (r.randint(1, 32),)
            kw = {"elementwise_affine": r.random() < 0.5}
        else:
            a = (r.randint(1, 100), r.randint(1, 16))
            kw = {}
        src = HDR + "nn.%s(%s)" % (kind, ", ".join([str(x) for x in a] + ["%s=%r" % kv for kv in kw.items()]))
        real = sum(p.numel() for p in getattr(nn, kind)(*a, **kw).parameters())
        assert param_count(first_call(src)) == real, src


def test_sequential_ordered_dict():
    src = HDR + "from collections import OrderedDict\nnn.Sequential(OrderedDict([\n    ('conv', nn.Conv2d(1, 4, 3)),\n    ('act', nn.ReLU()),\n]))"
    layer = first_call(src)
    assert [c.kind for c in layer.children] == ["Conv2d", "ReLU"]
    assert [c.name for c in layer.children] == ["Conv2d", "ReLU"]
    assert layer.child_names == ["conv", "act"]
