import ast
import textwrap
import time

import pytest

from deepfriend.shapes.extract import safe_source
from deepfriend.shapes.static_torch import analyze_torch

torch = pytest.importorskip("torch")

from deepfriend.shapes.runtime_torch import Worker  # noqa: E402


@pytest.fixture(scope="module")
def worker():
    w = Worker()
    yield w
    w.close()


def src_of(s):
    return textwrap.dedent(s).lstrip("\n")


HEADER = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"


def ln(src, needle, nth=0):
    return [i for i, l in enumerate(src.splitlines()) if needle in l][nth]


def test_safe_source_keeps_only_allowed(tmp_path):
    src = src_of("""
        import os
        import torch
        import requests
        from torch import nn
        import numpy as np
        LR = 0.1
        DATA = load_data()
        print("hello")
        open("x.txt", "w")
        @some_decorator
        def helper(c):
            return nn.Conv2d(c, c, 3)
        class Net(nn.Module):
            x = side_effect()
            def forward(self, x):
                return x
        if __name__ == "__main__":
            train()
    """)
    mod = safe_source(ast.parse(src))
    kept = ast.unparse(mod)
    assert "import torch" in kept and "from torch import nn" in kept and "import numpy as np" in kept
    assert "import os" not in kept and "requests" not in kept
    assert "LR = 0.1" in kept
    assert "load_data" not in kept and "print" not in kept and "open(" not in kept and "train()" not in kept
    assert "def helper" in kept and "some_decorator" not in kept
    assert "side_effect" not in kept and "def forward" in kept
    # line numbers are preserved
    net = [n for n in mod.body if isinstance(n, ast.ClassDef)][0]
    assert net.lineno - 1 == ln(src, "class Net")


def test_top_level_code_never_runs(worker, tmp_path):
    marker = tmp_path / "pwned.txt"
    src = HEADER + src_of("""
        open(%r, "w").write("x")
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)
            def forward(self, x):
                return self.fc(x)
        train_forever()
    """ % str(marker))
    res = worker.check(src, "M", [(3, 4)])
    assert res is not None and res["error"] is None
    assert not marker.exists()


def test_side_effects_inside_model_are_blocked(worker, tmp_path):
    marker = tmp_path / "inside.txt"
    src = HEADER + src_of("""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                open(%r, "w").write("x")
                self.fc = nn.Linear(4, 2)
            def forward(self, x):
                return self.fc(x)
    """ % str(marker))
    assert worker.check(src, "M", [(3, 4)]) is None
    assert not marker.exists()


def test_network_blocked(worker):
    src = HEADER + src_of("""
        import urllib.request
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                import urllib.request as u
                u.urlopen("http://example.com")
                self.fc = nn.Linear(4, 2)
            def forward(self, x):
                return self.fc(x)
    """)
    t = time.time()
    assert worker.check(src, "M", [(3, 4)]) is None
    assert time.time() - t < 3


FIXTURES = [
    ("lenet", (2, 1, 32, 32), """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""),
    ("vgg", (3, 3, 32, 32), """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.head = nn.Linear(128, 5)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)
"""),
    ("mlp", (4, 28, 28), """
class Model(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 10))

    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=1)
        return probs
"""),
    ("conv1d", (2, 8, 100), """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(8, 16, 5, stride=2, padding=2)
        self.p = nn.MaxPool1d(3)
        self.bn = nn.BatchNorm1d(16)

    def forward(self, x):
        y = self.bn(self.c(x))
        y = self.p(y)
        z = y.mean(dim=2)
        return z
"""),
    ("branches", (2, 3, 16, 16), """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 4, 1)
        self.b = nn.Conv2d(3, 4, 3, padding=1)
        self.out = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        ya = self.a(x)
        yb = self.b(x)
        y = torch.cat([ya, yb], dim=1)
        y = self.out(y)
        return y.permute(0, 2, 3, 1)
"""),
    ("embed", (2, 7), """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 12)
        self.norm = nn.LayerNorm(12)
        self.fc = nn.Linear(12, 3)

    def forward(self, x):
        e = self.norm(self.emb(x))
        pooled = e.mean(dim=1)
        return self.fc(pooled)
"""),
]


@pytest.mark.parametrize("name,shape,body", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_runtime_matches_static_and_real(worker, name, shape, body):
    src = HEADER + "# df: input %s\n" % (shape,) + body.lstrip("\n")
    res = worker.check(src, "Model", [shape])
    assert res is not None, "runtime check failed"
    assert res["error"] is None
    # 1) runtime per-line shapes agree with the independent static analyzer
    static = analyze_torch(ast.parse(src), src)[0]
    runtime_lines = {line: shapes for line, shapes in res["lines"]}
    for h in static["hints"]:
        if h["line"] in runtime_lines and h["shape"] != "?":
            assert runtime_lines[h["line"]][0] == h["shape"], (name, h, runtime_lines[h["line"]])
    # 2) runtime module outputs agree with a real CPU forward pass
    ns = {}
    exec(compile(src, "<real>", "exec"), ns)
    real_model = ns["Model"]().eval()
    seen = {}
    for mname, mod in real_model.named_modules():
        if mname:
            mod.register_forward_hook(lambda m, i, o, mname=mname: seen.setdefault(mname, []).append(str(tuple(o.shape))))
    x = torch.zeros(*shape, dtype=torch.long) if name == "embed" else torch.zeros(*shape)
    with torch.no_grad():
        real_model(x)
    for mname, outs in seen.items():
        assert res["modules"][mname]["out"][0] == outs[0], (name, mname)
    assert res["totalParams"] == sum(p.numel() for p in real_model.parameters())


def test_custom_op_becomes_known(worker):
    src = HEADER + src_of("""
        # df: input (2, 5)
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(1))
                self.fc = nn.Linear(5, 3)

            def forward(self, x):
                x = x * self.scale
                x = x @ torch.ones(5, 5)
                return self.fc(x)
    """)
    res = worker.check(src, "M", [(2, 5)])
    lines = dict((l, s) for l, s in res["lines"])
    assert lines[ln(src, "x = x * self.scale")] == ["(2, 5)"]
    assert lines[ln(src, "x = x @ torch")] == ["(2, 5)"]
    assert lines[ln(src, "return self.fc")] == ["(2, 3)"]


def test_shape_error_reported_with_line(worker):
    src = HEADER + src_of("""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(7, 3))

            def forward(self, x):
                y = x @ self.w
                return y
    """)
    res = worker.check(src, "M", [(2, 5)])
    assert res is not None
    assert res["error"]["line"] == ln(src, "y = x @ self.w")
    assert "reduction dim" in res["error"]["message"]


def test_ctor_with_required_args_skipped(worker):
    src = HEADER + "class M(nn.Module):\n    def __init__(self, n):\n        super().__init__()\n        self.fc = nn.Linear(n, 2)\n    def forward(self, x):\n        return self.fc(x)\n"
    assert worker.check(src, "M", [(2, 3)]) is None


def test_huge_tensor_in_init_is_free_on_meta(worker):
    src = HEADER + src_of("""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.big = nn.Parameter(torch.randn(10**10))
                self.fc = nn.Linear(4, 2)
            def forward(self, x):
                return self.fc(x)
    """)
    t = time.time()
    res = worker.check(src, "M", [(1, 4)])
    assert res is not None and res["totalParams"] == 10**10 + 10
    assert time.time() - t < 3


def test_runtime_timeout_falls_back(worker):
    src = HEADER + src_of("""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                while True:
                    pass
            def forward(self, x):
                return x
    """)
    t = time.time()
    assert worker.check(src, "M", [(1, 4)], timeout=1.5) is None
    assert time.time() - t < 4
    ok = HEADER + "class N(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 2)\n    def forward(self, x):\n        return self.fc(x)\n"
    res = worker.check(ok, "N", [(1, 4)])
    assert res is not None and res["lines"]


def test_symbolic_dims_substituted(worker):
    src = HEADER + "class M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 2)\n    def forward(self, x):\n        return self.fc(x)\n"
    res = worker.check(src, "M", [("B", 4)])
    assert dict((l, s) for l, s in res["lines"])[ln(src, "return self.fc")] == ["(2, 2)"]



@pytest.mark.parametrize("body,needle", [
    ("self.fc = nn.Linear(10, 3)", "return self.fc(x)"),
], ids=["linear"])
def test_meta_linear_mismatch_is_shape_error(worker, body, needle):
    src = HEADER + "class M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        %s\n    def forward(self, x):\n        return self.fc(x)\n" % body
    res = worker.check(src, "M", [(2, 12)])
    assert res is not None and res["error"]["line"] == ln(src, needle)


def test_meta_conv_channels_is_shape_error(worker):
    src = HEADER + "class M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.c = nn.Conv2d(3, 4, 3)\n    def forward(self, x):\n        return self.c(x)\n"
    res = worker.check(src, "M", [(1, 1, 8, 8)])
    assert res is not None and res["error"]["line"] == ln(src, "return self.c(x)")
