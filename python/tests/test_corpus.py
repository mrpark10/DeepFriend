"""Acceptance: on realistic models, every shape the static analyzer shows equals real PyTorch."""
import ast
import os

import pytest

from deepfriend.shapes.static_torch import analyze_torch

torch = pytest.importorskip("torch")
from deepfriend.shapes.runtime_torch import Worker  # noqa: E402

SRC_PATH = os.path.join(os.path.dirname(__file__), "corpus", "real_models.py")


@pytest.fixture(scope="module")
def setup():
    text = open(SRC_PATH, encoding="utf-8").read()
    w = Worker()
    yield text, analyze_torch(ast.parse(text), text), w
    w.close()


NAMES = ["TinyResNet", "TinyUNet", "VGGLoop", "TextCNN", "AutoEncoder", "WithAttention"]


@pytest.mark.parametrize("name", NAMES)
def test_static_shapes_equal_real_pytorch(setup, name):
    text, models, worker = setup
    m = next(x for x in models if x["name"] == name)
    assert m["diagnostics"] == [], m["diagnostics"]
    shapes = [tuple(int(d) for d in s.strip("()").split(",") if d.strip()) for s in m["input"]["shapes"]]
    rt = worker.check(text, name, shapes)
    assert rt is not None and rt["error"] is None, rt
    real = {line: s for line, s in rt["lines"]}
    known = [h for h in m["hints"] if h["shape"] != "?"]
    compared = 0
    for h in m["hints"]:
        if h["shape"] == "?" or h["line"] not in real:
            continue
        assert real[h["line"]][0] == h["shape"], (name, h, real[h["line"]], text.splitlines()[h["line"]])
        compared += 1
    assert compared >= max(1, len(known) // 2), (name, compared, len(known))


def test_attention_line_shows_output_and_weights(setup):
    text, models, _ = setup
    m = next(x for x in models if x["name"] == "WithAttention")
    attn_line = next(i for i, l in enumerate(text.splitlines()) if "a, _ = self.attn" in l)
    h = {x["line"]: x for x in m["hints"]}
    assert h[attn_line]["shape"] == "(2, 10, 32), (2, 10, 10)"


def test_resnet_params_match_real(setup):
    text, models, _ = setup
    ns = {}
    exec(compile(text, "<corpus>", "exec"), ns)
    for name in NAMES:
        m = next(x for x in models if x["name"] == name)
        if m["totalParams"] is not None:
            real = sum(p.numel() for p in ns[name]().parameters())
            assert m["totalParams"] == real, (name, m["totalParams"], real)
