"""Static Keras results must match real Keras for random Sequential models."""
import ast
import os
import random

import pytest

from deepfriend.shapes.static_keras import analyze_keras

keras = pytest.importorskip("keras")

N = int(os.environ.get("DF_PARITY_N", "150"))


def random_model(r):
    lines = ["import keras", "from keras import layers", "model = keras.Sequential(["]
    h, w = r.randint(4, 40), r.randint(4, 40)
    c = r.randint(1, 4)
    lines.append("    keras.Input(shape=(%d, %d, %d))," % (h, w, c))
    flat = False
    for _ in range(r.randint(1, 6)):
        choice = r.choice(["conv", "conv", "pool", "avg", "dense", "flatten", "bn", "drop", "gap"])
        if choice == "conv":
            k = r.choice([1, 2, 3, 5, "(3, 1)"])
            s = r.choice([1, 1, 2, "(2, 1)"])
            pad = r.choice(["valid", "same"])
            d = r.choice([1, 1, 2]) if s == 1 else 1
            lines.append("    layers.Conv2D(%d, %s, strides=%s, padding=%r, dilation_rate=%d)," % (r.randint(1, 8), k, s, pad, d))
        elif choice in ("pool", "avg"):
            name = "MaxPooling2D" if choice == "pool" else "AveragePooling2D"
            ps = r.choice([2, 3, "(2, 1)"])
            st = r.choice(["None", 1, 2])
            pad = r.choice(["valid", "same"])
            lines.append("    layers.%s(pool_size=%s, strides=%s, padding=%r)," % (name, ps, st, pad))
        elif choice == "dense":
            lines.append("    layers.Dense(%d, use_bias=%s)," % (r.randint(1, 8), r.choice([True, False])))
        elif choice == "flatten":
            lines.append("    layers.Flatten(),")
            flat = True
        elif choice == "bn":
            lines.append("    layers.BatchNormalization(),")
        elif choice == "gap":
            lines.append("    layers.GlobalAveragePooling2D(),")
        else:
            lines.append("    layers.Dropout(0.3),")
    lines.append("])")
    return "\n".join(lines) + "\n", flat


def test_keras_parity():
    r = random.Random(11)
    mismatches = []
    for _ in range(N):
        src, _ = random_model(r)
        models = analyze_keras(ast.parse(src), src)
        ours = models[0]
        ns = {}
        try:
            exec(compile(src, "<gen>", "exec"), ns)
            real_model = ns["model"]
            real = [str(tuple(l.output.shape)) for l in real_model.layers]
            real_params = real_model.count_params()
            raised = False
        except Exception:
            raised = True
        if raised:
            if not ours["diagnostics"]:
                mismatches.append(("keras raised, we did not", src))
            continue
        if ours["diagnostics"]:
            mismatches.append(("we raised, keras did not", src, ours["diagnostics"]))
            continue
        mine = [b["outShape"] for b in ours["blocks"]]
        if mine != real or ours["totalParams"] != real_params:
            mismatches.append((src, mine, real, ours["totalParams"], real_params))
    assert not mismatches, "%d mismatches, first: %r" % (len(mismatches), mismatches[0])
