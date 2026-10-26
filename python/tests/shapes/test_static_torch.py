import ast
import textwrap

from deepfriend.shapes.annotations import find_input, suggest_input
from deepfriend.shapes.static_torch import analyze_torch


def run(src):
    src = textwrap.dedent(src).lstrip("\n")
    return src, analyze_torch(ast.parse(src), src)


def ln(src, needle, nth=0):
    hits = [i for i, l in enumerate(src.splitlines()) if needle in l]
    return hits[nth]


def hints(model):
    return {h["line"]: h["shape"] for h in model["hints"]}


def model(models, name):
    return next(m for m in models if m["name"] == name)


HEADER = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n"


def test_spec_example_resnet_stem():
    src, models = run(HEADER + """
# df: input (32, 3, 224, 224)
class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
""")
    m = model(models, "Stem")
    h = hints(m)
    assert h[ln(src, "self.conv = ")] == "(32, 64, 112, 112)"
    assert h[ln(src, "x = self.conv(x)")] == "(32, 64, 112, 112)"
    assert h[ln(src, "x = self.pool(x)")] == "(32, 64, 56, 56)"
    assert h[ln(src, "self.pool = ")] == "(32, 64, 56, 56)"
    assert h[ln(src, "return x")] == "(32, 64, 56, 56)"
    assert m["input"] == {"line": ln(src, "# df: input"), "shapes": ["(32, 3, 224, 224)"]}
    assert m["needsInput"] is False
    assert m["diagnostics"] == []
    assert m["kind"] == "torch"
    assert m["line"] == ln(src, "class Stem")


CNN_BUG = HEADER + """
# df: input (64, 1, 28, 28)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(10000, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""


def test_linear_mismatch_diag_and_fix():
    src, models = run(CNN_BUG)
    m = model(models, "Net")
    h = hints(m)
    assert h[ln(src, "x = x.view")] == "(64, 3136)"
    diags = m["diagnostics"]
    defs = [d for d in diags if d["line"] == ln(src, "self.fc1 =")]
    calls = [d for d in diags if d["line"] == ln(src, "F.relu(self.fc1(x))")]
    assert len(defs) == 1 and len(calls) == 1
    d = defs[0]
    assert d["code"] == "linear_in" and d["args"] == {"expected": 10000, "actual": 3136}
    line_text = src.splitlines()[d["line"]].encode()
    assert line_text[d["col"]:d["endCol"]] == b"10000"
    assert d["fix"]["newText"] == "3136"
    assert line_text[d["fix"]["col"]:d["fix"]["endCol"]] == b"10000"
    assert calls[0]["fix"] == d["fix"]


def test_no_cascade_after_error():
    src, models = run(CNN_BUG)
    m = model(models, "Net")
    assert len({(d["code"]) for d in m["diagnostics"]}) == 1
    h = {x["line"]: x for x in m["hints"]}
    after = h[ln(src, "x = self.fc2(x)")]
    assert after["shape"] == "?" and after["unknownReason"] == "after_error"


def test_sequential_children_hints():
    src, models = run(HEADER + """
# df: input (8, 3, 32, 32)
class VGGish(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))
""")
    m = model(models, "VGGish")
    h = hints(m)
    assert h[ln(src, "nn.Conv2d(3, 16")] == "(8, 16, 32, 32)"
    assert h[ln(src, "nn.MaxPool2d(2)", 0)] == "(8, 16, 16, 16)"
    assert h[ln(src, "nn.MaxPool2d(2)", 1)] == "(8, 32, 8, 8)"
    assert h[ln(src, "self.classifier =")] == "(8, 10)"
    assert h[ln(src, "return self.classifier")] == "(8, 10)"
    assert [b["name"] for b in m["blocks"]][:3] == ["features.0", "features.1", "features.2"]
    assert {k: v for k, v in m["blocks"][-1].items() if k != "_hintLine"} == {"name": "classifier.1", "kind": "Linear", "line": ln(src, "self.classifier ="), "outShape": "(8, 10)", "params": 32 * 64 * 10 + 10, "error": False}


def test_functional_and_view_variants():
    src, models = run(HEADER + """
# df: input (4, 3, 16, 16)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        a = torch.flatten(x, 1)
        b = x.reshape(-1, 8 * 7 * 7)
        c = x.flatten(start_dim=1)
        n = x.shape[0]
        d = x.view(n, -1)
        e = x.permute(0, 2, 3, 1)
        f = x.mean(dim=(2, 3))
        g = torch.cat([a, b], dim=1)
        y = a + b
        z = torch.softmax(a, dim=1)
        return x.unsqueeze(1)
""")
    h = hints(model(models, "M"))
    assert h[ln(src, "x = F.max_pool2d")] == "(4, 8, 7, 7)"
    for name in ["a = ", "b = ", "c = ", "d = ", "y = ", "z = "]:
        assert h[ln(src, "        " + name)] == "(4, 392)", name
    assert h[ln(src, "e = x.permute")] == "(4, 7, 7, 8)"
    assert h[ln(src, "f = x.mean")] == "(4, 8)"
    assert h[ln(src, "g = torch.cat")] == "(4, 784)"
    assert h[ln(src, "return x.unsqueeze")] == "(4, 1, 8, 7, 7)"
    assert ln(src, "n = x.shape[0]") not in h


def test_cat_two_branches_mismatch():
    src, models = run(HEADER + """
# df: input (2, 3, 8, 8)
class TwoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 4, 3, padding=1)
        self.b = nn.Conv2d(3, 4, 3)

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], dim=1)
""")
    m = model(models, "TwoBranch")
    assert [d["code"] for d in m["diagnostics"]] == ["cat_mismatch"]
    assert m["diagnostics"][0]["line"] == ln(src, "return torch.cat")


def test_modulelist_loop_and_listcomp():
    src, models = run(HEADER + """
# df: input (5, 16)
class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(3)])
        self.out = nn.Linear(16, 2)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)
""")
    m = model(models, "Deep")
    h = hints(m)
    assert h[ln(src, "x = F.relu(layer(x))")] == "(5, 16)"
    assert h[ln(src, "return self.out")] == "(5, 2)"
    assert m["totalParams"] == 3 * (16 * 16 + 16) + 16 * 2 + 2


def test_nested_user_module():
    src, models = run(HEADER + """
class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# df: input (2, 3, 32, 32)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = Block(3, 16)
        self.b2 = Block(16, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.b2(self.b1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)
""")
    net = model(models, "Net")
    h = hints(net)
    assert h[ln(src, "x = self.b2(self.b1(x))")] == "(2, 32, 32, 32)"
    assert h[ln(src, "x = self.pool(x)")] == "(2, 32)"
    blk = model(models, "Block")
    assert blk["needsInput"] is False
    bh = hints(blk)
    assert bh[ln(src, "return F.relu(self.bn")] == "(2, 16, 32, 32)"
    assert net["totalParams"] == (3 * 16 * 9 + 16 + 32) + (16 * 32 * 9 + 32 + 64) + 32 * 10 + 10


def test_helper_method_inlined():
    src, models = run(HEADER + """
# df: input (2, 10)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 4)

    def encode(self, x):
        return self.fc(x)

    def forward(self, x):
        z = self.encode(x)
        return z
""")
    h = hints(model(models, "M"))
    assert h[ln(src, "z = self.encode(x)")] == "(2, 4)"


def test_dm_alias_comment_and_decorator_gap():
    src, models = run(HEADER + """
# dm: input (1, 4)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)
""")
    assert hints(model(models, "M"))[ln(src, "return self.fc")] == "(1, 2)"


def test_multiple_inputs():
    src, models = run(HEADER + """
# df: input (3, 5), (3, 7)
class Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 1)

    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)
        return self.fc(x)
""")
    m = model(models, "Fuse")
    assert hints(m)[ln(src, "x = torch.cat")] == "(3, 12)"
    assert m["input"]["shapes"] == ["(3, 5)", "(3, 7)"]


def test_symbolic_batch():
    src, models = run(HEADER + """
# df: input (B, 1, 28, 28)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.fc = nn.Linear(4 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""")
    m = model(models, "M")
    h = hints(m)
    assert h[ln(src, "x = x.view")] == "(B, 2704)"
    assert h[ln(src, "return self.fc")] == "(B, 10)"
    assert m["memory"] is None
    assert m["totalParams"] == 4 * 9 + 4 + 2704 * 10 + 10


def test_unknown_ctor_arg_is_question():
    src, models = run(HEADER + """
# df: input (2, 8)
class M(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(n, 10)
        self.out = nn.Linear(10, 3)

    def forward(self, x):
        x = self.fc(x)
        return self.out(x)
""")
    m = model(models, "M")
    h = {x["line"]: x for x in m["hints"]}
    assert h[ln(src, "x = self.fc(x)")]["shape"] == "?"
    assert h[ln(src, "x = self.fc(x)")]["unknownReason"] == "ctor_arg"
    assert m["diagnostics"] == []
    assert m["totalParams"] is None


def test_if_branches_disagree_unknown():
    src, models = run(HEADER + """
# df: input (2, 8)
class M(nn.Module):
    def __init__(self, use_big=False):
        super().__init__()
        self.small = nn.Linear(8, 4)
        self.big = nn.Linear(8, 16)
        self.use_big = use_big

    def forward(self, x):
        if self.training:
            y = self.small(x)
        else:
            y = self.big(x)
        z = x * 2
        return y
""")
    h = {x["line"]: x for x in model(models, "M")["hints"]}
    assert h[ln(src, "return y")]["shape"] == "?"
    assert h[ln(src, "return y")]["unknownReason"] == "branch"
    assert h[ln(src, "y = self.small(x)")]["shape"] == "(2, 4)"
    assert h[ln(src, "z = x * 2")]["shape"] == "(2, 8)"


def test_if_branches_agree_known():
    src, models = run(HEADER + """
# df: input (2, 8)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 4)
        self.b = nn.Linear(8, 4)

    def forward(self, x):
        if x.sum() > 0:
            y = self.a(x)
        else:
            y = self.b(x)
        return y
""")
    assert hints(model(models, "M"))[ln(src, "return y")] == "(2, 4)"


def test_unsupported_op_question_and_continues():
    src, models = run(HEADER + """
# df: input (2, 3, 8, 8)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = nn.MultiheadAttention(8, 2)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        y = F.interpolate(x, scale_factor=2)
        z = self.fc(x)
        return z
""")
    h = {x["line"]: x for x in model(models, "M")["hints"]}
    assert h[ln(src, "y = F.interpolate")]["shape"] == "?"
    assert h[ln(src, "y = F.interpolate")]["unknownReason"].startswith("unsupported_op")
    assert h[ln(src, "z = self.fc(x)")]["shape"] == "(2, 3, 8, 2)"


def test_conv_in_channels_fix():
    src, models = run(HEADER + """
# df: input (1, 1, 28, 28)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)

    def forward(self, x):
        return self.conv(x)
""")
    d = [d for d in model(models, "M")["diagnostics"] if d["line"] == ln(src, "self.conv =")][0]
    assert d["code"] == "conv_in" and d["fix"]["newText"] == "1"


def test_no_fix_when_arg_uses_names():
    src, models = run(HEADER + """
HIDDEN = 100
# df: input (1, 50)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN, 2)

    def forward(self, x):
        return self.fc(x)
""")
    d = [d for d in model(models, "M")["diagnostics"] if d["line"] == ln(src, "self.fc =")][0]
    assert d["args"] == {"expected": 100, "actual": 50}
    assert "fix" not in d


def test_needs_input_and_suggestion():
    src, models = run(HEADER + """
from torchvision import datasets
train = datasets.MNIST("data", download=True)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))
""")
    m = model(models, "M")
    assert m["needsInput"] is True
    assert m["suggestedInput"] == "(B, 1, 28, 28)"
    assert m["hints"] == []
    assert [{k: v for k, v in b.items() if k != "_hintLine"} for b in m["blocks"]] == [{"name": "fc", "kind": "Linear", "line": ln(src, "self.fc ="), "outShape": "?", "params": 7850, "error": False}]


LENET = HEADER + """
# df: input (1, 1, 32, 32)
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""


def test_params_and_memory():
    src, models = run(LENET)
    m = model(models, "LeNet")
    assert m["totalParams"] == 61706
    assert [b["name"] for b in m["blocks"]] == ["conv1", "pool", "conv2", "pool", "fc1", "fc2", "fc3"]
    assert m["memory"] == {"params": 246824, "grads": 246824, "optimizer": 493648, "activations": 32376, "total": 1019672}
    assert hints(m)[ln(src, "self.pool =")] == "(1, 6, 14, 14)"
    assert [x for x in m["hints"] if x["line"] == ln(src, "self.pool =")][0]["all"] == ["(1, 6, 14, 14)", "(1, 16, 5, 5)"]


def test_multiline_statement_hint_on_last_line():
    src, models = run(HEADER + """
# df: input (2, 4)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(
            4,
            3,
        )

    def forward(self, x):
        y = self.fc(
            x
        )
        return y
""")
    h = hints(model(models, "M"))
    assert h[ln(src, "        )", 1)] == "(2, 3)"
    assert h[ln(src, "        )", 0)] == "(2, 3)"


def test_find_input_variants():
    lines = ["import torch", "", "#   df:input  ( 2 , 3 )", "@something", "class A:"]
    spec = find_input(lines, 4)
    assert spec.line == 2 and spec.shapes == [(2, 3)]
    assert find_input(["x = 1", "class A:"], 1) is None
    assert find_input(["# df: input (N, 3), (N,)", "class A:"], 1).shapes == [("N", 3), ("N",)]
    assert find_input(["# df: input oops", "class A:"], 1) is None
    assert find_input(["# df: input (2, 3)", "x = 5", "class A:"], 2) is None


def test_suggest_input():
    assert suggest_input("datasets.CIFAR10(root)") == "(B, 3, 32, 32)"
    assert suggest_input("transforms.Resize(224)") == "(B, 3, 224, 224)"
    assert suggest_input("FashionMNIST") == "(B, 1, 28, 28)"
    assert suggest_input("nothing here") is None


def test_non_module_classes_ignored():
    _, models = run(HEADER + "class Config:\n    lr = 0.1\n")
    assert models == []
