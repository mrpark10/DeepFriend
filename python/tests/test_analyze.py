import json
import statistics
import textwrap
import time

import pytest

from deepfriend.shapes.analyze import Analyzer, merge, to_utf16_col

HEADER = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"

GOOD = HEADER + textwrap.dedent("""
# df: input (2, 4)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        y = self.fc(x)
        return y
""")


def ln(src, needle):
    return [i for i, l in enumerate(src.splitlines()) if needle in l][0]


def test_basic_result_shape():
    res = Analyzer().analyze("file:///a.py", GOOD, runtime=False)
    assert res["framework"] == "pytorch"
    assert res["syntaxError"] is None
    m = res["models"][0]
    assert {h["line"]: h["shape"] for h in m["hints"]}[ln(GOOD, "y = self.fc")] == "(2, 3)"


def test_framework_detection():
    a = Analyzer()
    assert a.analyze("u1", "import tensorflow as tf\n", False)["framework"] == "tensorflow"
    assert a.analyze("u2", "from keras import layers\n", False)["framework"] == "tensorflow"
    assert a.analyze("u3", "x = 1\n", False)["framework"] is None


def test_syntax_error_keeps_previous():
    a = Analyzer()
    first = a.analyze("file:///a.py", GOOD, runtime=False)
    broken = GOOD.replace("return y", "return y +")
    res = a.analyze("file:///a.py", broken, runtime=False)
    assert res["syntaxError"] is not None and res["syntaxError"]["line"] == ln(broken, "return y +")
    assert res["models"] == first["models"]
    other = a.analyze("file:///other.py", "def (", runtime=False)
    assert other["models"] == [] and other["syntaxError"] is not None


def test_cache_returns_same_object_quickly():
    a = Analyzer()
    r1 = a.analyze("u", GOOD, False)
    r2 = a.analyze("u", GOOD, False)
    assert r1 == r2


def test_to_utf16_col():
    line = 'self.tag = "🚀"; x = 1'
    byte_col = line.encode("utf-8").index(b"x = 1")
    assert to_utf16_col(line, byte_col) == line.index("x = 1") + 1  # emoji counts as 2 UTF-16 units


def test_korean_identifiers_utf16_cols():
    src = HEADER + textwrap.dedent("""
    # df: input (1, 50)
    class 모델(nn.Module):
        def __init__(self):
            super().__init__()
            self.이름 = "🚀"; self.층 = nn.Linear(10000, 10)

        def forward(self, x):
            return self.층(x)
    """)
    res = Analyzer().analyze("u", src, False)
    m = res["models"][0]
    assert m["name"] == "모델"
    d = [d for d in m["diagnostics"] if d["line"] == ln(src, "self.층 =")][0]
    line = src.splitlines()[d["line"]]
    py_idx = line.index("10000")
    assert d["col"] == py_idx + 1  # +1 for the astral emoji
    assert d["endCol"] == py_idx + 1 + 5
    assert d["fix"]["col"] == d["col"]


def test_analyzer_never_raises_on_weird_code():
    a = Analyzer()
    weird = HEADER + "class A(nn.Module):\n    def forward(self, *args, **kw):\n        return args[0]\n# df: input (1,)\nclass B(A):\n    pass\n"
    res = a.analyze("u", weird, False)
    assert isinstance(res["models"], list)


def test_perf_static_1000_lines():
    parts = [HEADER]
    for i in range(30):
        parts.append(textwrap.dedent("""
        # df: input (8, 3, 32, 32)
        class Net%d(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.fc1 = nn.Linear(32 * 8 * 8, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, 0.5, self.training)
                return self.fc2(x)
        """ % i))
    src = "".join(parts)
    assert len(src.splitlines()) >= 650
    src = src + "\n" * (1000 - len(src.splitlines()))
    times = []
    for k in range(5):
        a = Analyzer()
        t = time.perf_counter()
        res = a.analyze("u%d" % k, src, False)
        times.append(time.perf_counter() - t)
    assert len(res["models"]) == 30
    assert statistics.median(times) < 0.1, times


# ---- merge -----------------------------------------------------------------------

def _model(hints, diags=(), blocks=()):
    return {"name": "M", "kind": "torch", "line": 0, "endLine": 50, "input": {"line": 0, "shapes": ["(2, 4)"]},
            "needsInput": False, "suggestedInput": None, "hints": list(hints), "diagnostics": list(diags),
            "blocks": list(blocks), "totalParams": None, "memory": None, "verified": False}


def test_merge_runtime_overrides_and_fills():
    m = _model([{"line": 5, "shape": "?", "unknownReason": "unsupported_op:x"}, {"line": 6, "shape": "(2, 3)"}],
               blocks=[{"name": "fc", "kind": "Linear", "line": 3, "outShape": "?", "params": None, "error": False, "_hintLine": 3}])
    rt = {"lines": [[5, ["(2, 9)"]], [7, ["(2, 1)"]]], "modules": {"fc": {"out": ["(2, 3)"], "params": 15}}, "totalParams": 15, "error": None}
    out = merge([m], rt, symbolic=False, lines=["x"] * 60, owner=m)
    h = {x["line"]: x for x in out[0]["hints"]}
    assert h[5] == {"line": 5, "shape": "(2, 9)"}
    assert h[6]["shape"] == "(2, 3)"
    assert h[7]["shape"] == "(2, 1)"
    assert h[3]["shape"] == "(2, 3)"
    assert out[0]["blocks"][0]["outShape"] == "(2, 3)" and out[0]["blocks"][0]["params"] == 15
    assert out[0]["totalParams"] == 15
    assert out[0]["verified"] is True


def test_merge_runtime_ok_removes_static_false_alarm():
    m = _model([], diags=[{"line": 4, "col": 0, "endLine": 4, "endCol": 3, "code": "linear_in", "args": {}}])
    rt = {"lines": [], "modules": {}, "totalParams": 1, "error": None}
    assert merge([m], rt, symbolic=False, lines=["abc"] * 10, owner=m)[0]["diagnostics"] == []


def test_merge_runtime_error_added_when_static_silent():
    m = _model([])
    rt = {"lines": [], "modules": {}, "totalParams": 1, "error": {"line": 8, "message": "a and b must have same reduction dim", "type": "RuntimeError"}}
    lines = ["        y = x @ self.w"] * 10
    d = merge([m], rt, symbolic=False, lines=lines, owner=m)[0]["diagnostics"]
    assert d == [{"line": 8, "col": 8, "endLine": 8, "endCol": len(lines[8]), "code": "runtime_error", "args": {"message": "a and b must have same reduction dim"}}]


def test_merge_symbolic_only_errors():
    m = _model([{"line": 5, "shape": "(B, 3)"}])
    rt = {"lines": [[5, ["(2, 3)"]]], "modules": {}, "totalParams": 1, "error": None}
    out = merge([m], rt, symbolic=True, lines=["x"] * 10, owner=m)
    assert out[0]["hints"][0]["shape"] == "(B, 3)"
    assert out[0]["verified"] is False


def test_runtime_merge_marks_verified_end_to_end():
    pytest.importorskip("torch")
    a = Analyzer()
    try:
        res = a.analyze("u", GOOD, runtime=True)
    finally:
        a.close()
    assert res["models"][0]["verified"] is True


# ---- server: latest wins ------------------------------------------------------------


def test_latest_wins_cancels_older(monkeypatch):
    """Deterministic: the first analyze blocks; the second is superseded by the third."""
    import io
    import threading

    from deepfriend import server as srv
    from deepfriend.shapes import analyze as an

    started = threading.Event()
    release = threading.Event()
    calls = []

    class SlowAnalyzer:
        def analyze(self, uri, text, runtime):
            calls.append(text)
            if len(calls) == 1:
                started.set()
                release.wait(5)
            return {"framework": None, "models": [], "syntaxError": None, "text": text}

        def close(self):
            pass

    monkeypatch.setattr(an, "Analyzer", SlowAnalyzer)
    out = io.BytesIO()
    s = srv.Server(stdin=io.BytesIO(), stdout=out)
    s.handle({"id": 1, "method": "analyze", "params": {"uri": "u", "text": "one"}})
    assert started.wait(5)
    s.handle({"id": 2, "method": "analyze", "params": {"uri": "u", "text": "two"}})
    s.handle({"id": 3, "method": "analyze", "params": {"uri": "u", "text": "three"}})
    release.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        msgs = [json.loads(l) for l in out.getvalue().decode().splitlines() if l.strip()]
        if len(msgs) == 3:
            break
        time.sleep(0.02)
    got = {m["id"]: m for m in msgs}
    assert got[2]["cancelled"] is True and got[2]["result"] is None
    assert got[1]["result"]["text"] == "one" and got[3]["result"]["text"] == "three"
    assert calls == ["one", "three"]


def test_analyze_requests_for_different_files_are_all_answered(monkeypatch):
    import io

    from deepfriend import server as srv
    from deepfriend.shapes import analyze as an

    class Fast:
        def analyze(self, uri, text, runtime):
            return {"uri": uri}

        def close(self):
            pass

    monkeypatch.setattr(an, "Analyzer", Fast)
    out = io.BytesIO()
    s = srv.Server(stdin=io.BytesIO(), stdout=out)
    for i, uri in enumerate(["a", "b", "c"]):
        s.handle({"id": i, "method": "analyze", "params": {"uri": uri, "text": ""}})
    deadline = time.time() + 5
    while time.time() < deadline and len(out.getvalue().splitlines()) < 3:
        time.sleep(0.02)
    msgs = [json.loads(l) for l in out.getvalue().decode().splitlines()]
    assert sorted(m["result"]["uri"] for m in msgs) == ["a", "b", "c"]
