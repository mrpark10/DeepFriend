"""LSTM/GRU/RNN/MultiheadAttention: static shapes and parameter counts equal real PyTorch."""
import ast
import random
import textwrap

import pytest

from deepfriend.shapes.static_torch import analyze_torch

torch = pytest.importorskip("torch")
nn = torch.nn


def analyze(src):
    src = textwrap.dedent(src).lstrip("\n")
    return src, analyze_torch(ast.parse(src), src)[0]


def test_rnn_family_parity():
    r = random.Random(3)
    for _ in range(60):
        kind = r.choice(["LSTM", "GRU", "RNN"])
        inp, hid, layers = r.randint(1, 6), r.randint(1, 6), r.randint(1, 3)
        bf = r.choice([True, False])
        bi = r.choice([True, False])
        n, L = r.randint(1, 3), r.randint(1, 5)
        shape = (n, L, inp) if bf else (L, n, inp)
        src, m = analyze(f"""
            import torch
            import torch.nn as nn
            # df: input {shape}
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.rnn = nn.{kind}({inp}, {hid}, num_layers={layers}, batch_first={bf}, bidirectional={bi})
                def forward(self, x):
                    out, h = self.rnn(x)
                    return out
        """)
        real_mod = getattr(nn, kind)(inp, hid, num_layers=layers, batch_first=bf, bidirectional=bi)
        real_out, _ = real_mod(torch.zeros(*shape))
        hints = {h["line"]: h["shape"] for h in m["hints"]}
        ret = [i for i, l in enumerate(src.splitlines()) if "return out" in l][0]
        assert hints[ret] == str(tuple(real_out.shape)).replace(",)", ",)"), (kind, shape, hints)
        assert m["totalParams"] == sum(p.numel() for p in real_mod.parameters()), (kind, inp, hid, layers, bi)


def test_rnn_input_size_mismatch_is_an_error_with_fix():
    src, m = analyze("""
        import torch
        import torch.nn as nn
        # df: input (4, 7, 10)
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(12, 32, batch_first=True)
            def forward(self, x):
                out, (h, c) = self.lstm(x)
                return out[:, -1]
    """)
    d = [d for d in m["diagnostics"] if d["line"] == [i for i, l in enumerate(src.splitlines()) if "nn.LSTM" in l][0]][0]
    assert d["code"] == "rnn_in" and d["args"] == {"expected": 12, "actual": 10}
    assert d["fix"]["newText"] == "10"


def test_lstm_hidden_tuple_unpacking_and_last_step():
    src, m = analyze("""
        import torch
        import torch.nn as nn
        # df: input (4, 7, 10)
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 32, num_layers=2, batch_first=True)
                self.fc = nn.Linear(32, 3)
            def forward(self, x):
                out, (h, c) = self.lstm(x)
                last = h[-1]
                return self.fc(last)
    """)
    hints = {h["line"]: h["shape"] for h in m["hints"]}
    line = lambda s: [i for i, l in enumerate(src.splitlines()) if s in l][0]
    assert hints[line("out, (h, c)")] == "(4, 7, 32), ((2, 4, 32), (2, 4, 32))" or hints[line("out, (h, c)")].startswith("(4, 7, 32)")
    assert hints[line("return self.fc")] == "(4, 3)"


def test_multihead_attention_parity():
    for bf in (True, False):
        shape = (2, 9, 16) if bf else (9, 2, 16)
        src, m = analyze(f"""
            import torch
            import torch.nn as nn
            # df: input {shape}
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.attn = nn.MultiheadAttention(16, 4, batch_first={bf})
                def forward(self, x):
                    a, w = self.attn(x, x, x)
                    return a
        """)
        real = nn.MultiheadAttention(16, 4, batch_first=bf)
        out, _ = real(*(torch.zeros(*shape),) * 3)
        hints = {h["line"]: h["shape"] for h in m["hints"]}
        ret = [i for i, l in enumerate(src.splitlines()) if "return a" in l][0]
        assert hints[ret] == str(tuple(out.shape))
        assert m["totalParams"] == sum(p.numel() for p in real.parameters())
