import json
import os
import subprocess
import sys
import textwrap

import pytest

from deepfriend.export.flutter import generate, notebook

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PKG = os.path.join(ROOT, "python")


def cli(python, args, cwd, timeout=600):
    env = dict(os.environ, PYTHONPATH=PKG, TF_CPP_MIN_LOG_LEVEL="3", PYTHONIOENCODING="utf-8")
    p = subprocess.run([python, "-m", "deepfriend.export.cli", *args], cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
    lines = [l for l in p.stdout.splitlines() if l.startswith("DEEPFRIEND_RESULT ")]
    assert lines, (p.stdout[-2000:], p.stderr[-3000:])
    return json.loads(lines[-1][len("DEEPFRIEND_RESULT "):]), p


# ---- flutter code generation ---------------------------------------------------------

def test_flutter_files_have_exact_shapes():
    files = generate("mnist_cnn", [1, 28, 28, 1], [1, 10], lang="ko", layout="NHWC")
    dart = files["lib/deepfriend_model.dart"]
    assert "static const List<int> inputShape = [1, 28, 28, 1];" in dart
    assert "static const List<int> outputShape = [1, 10];" in dart
    assert "Interpreter.fromAsset('assets/models/mnist_cnn.tflite')" in dart
    assert "package:tflite_flutter/tflite_flutter.dart" in dart
    pub = files["pubspec_snippet.yaml"]
    assert "tflite_flutter: ^0.12.1" in pub and "- assets/models/mnist_cnn.tflite" in pub
    assert "Flutter" in files["README.md"]


def test_flutter_mentions_channel_first_for_pytorch():
    dart = generate("net", [1, 3, 32, 32], [1, 10], lang="en", layout="NCHW")["lib/deepfriend_model.dart"]
    assert "channels first" in dart


def test_flutter_name_is_sanitized():
    files = generate("내 모델!", [1, 4], [1, 2], lang="ko", layout="NC")
    assert "assets/models/model.tflite" in files["pubspec_snippet.yaml"] or "assets/models/" in files["pubspec_snippet.yaml"]
    assert "!" not in files["pubspec_snippet.yaml"]


def test_colab_notebook_is_valid_json():
    nb = json.loads(notebook("Net", "model.py", "weights.pt", [1, 3, 32, 32], lang="ko"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "pip install litert-torch" in src and "litert_torch.convert" in src and "(1, 3, 32, 32)" in src
    assert nb["nbformat"] == 4


# ---- end to end -----------------------------------------------------------------------

def test_keras_export_end_to_end(tmp_path):
    pytest.importorskip("tensorflow")
    model_path = tmp_path / "m.keras"
    code = textwrap.dedent(f"""
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import keras
        from keras import layers
        m = keras.Sequential([keras.Input(shape=(28, 28, 1)), layers.Conv2D(4, 3, activation="relu"), layers.Flatten(), layers.Dense(10)])
        m.save({str(model_path)!r})
    """)
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=300)
    res, _ = cli(sys.executable, ["--keras", str(model_path), "--out", str(tmp_path / "out"), "--lang", "ko"], tmp_path)
    assert res["ok"] is True, res
    assert res["inputShape"] == [1, 28, 28, 1] and res["outputShape"] == [1, 10]
    assert res["verify"]["passed"] is True and res["verify"]["maxAbsDiff"] < 1e-3
    assert os.path.getsize(res["tflite"]) > 1000
    assert os.path.isfile(os.path.join(res["flutterDir"], "lib", "deepfriend_model.dart"))


EXPORT_PY = os.path.join(ROOT, ".venv-export", "bin", "python")


@pytest.mark.skipif(not os.path.exists(EXPORT_PY), reason="needs the .venv-export interpreter with litert-torch")
def test_torch_export_end_to_end(tmp_path):
    model_file = tmp_path / "model.py"
    model_file.write_text(textwrap.dedent("""
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 4, 3)
                self.fc = nn.Linear(4 * 26 * 26, 10)

            def forward(self, x):
                return self.fc(torch.relu(self.conv(x)).flatten(1))

        print("TRAINING SHOULD NOT RUN")
        open("side_effect.txt", "w").write("x")
    """))
    weights = tmp_path / "w.pt"
    subprocess.run([EXPORT_PY, "-c", f"import sys; sys.path.insert(0, {str(tmp_path)!r})\nimport torch, importlib.util\nspec = importlib.util.spec_from_file_location('m', {str(model_file)!r})\n" +
                    "src = open(spec.origin).read().split('print(')[0]\nns = {}\nexec(src, ns)\ntorch.save(ns['Net']().state_dict(), " + repr(str(weights)) + ")"], check=True, capture_output=True, timeout=300)
    res, p = cli(EXPORT_PY, ["--torch-file", str(model_file), "--class", "Net", "--weights", str(weights), "--input", "1,1,28,28", "--out", str(tmp_path / "out")], tmp_path)
    assert res["ok"] is True, (res, p.stderr[-2000:])
    assert res["inputShape"] == [1, 1, 28, 28] and res["outputShape"] == [1, 10]
    assert res["verify"]["passed"] is True
    assert "TRAINING SHOULD NOT RUN" not in p.stdout
    assert not (tmp_path / "side_effect.txt").exists()


PY39 = os.path.join(ROOT, ".venv39", "bin", "python")


@pytest.mark.skipif(not os.path.exists(PY39), reason="needs .venv39")
def test_torch_export_on_old_python_reports_platform_and_writes_colab(tmp_path):
    (tmp_path / "model.py").write_text("import torch\nclass Net(torch.nn.Module):\n    pass\n")
    res, _ = cli(PY39, ["--torch-file", str(tmp_path / "model.py"), "--class", "Net", "--input", "1,3,32,32", "--out", str(tmp_path / "out")], tmp_path)
    assert res["ok"] is False and res["error"]["code"] == "platform_unsupported"
    assert os.path.isfile(res["colab"])


@pytest.mark.skipif(not os.path.exists(PY39), reason="needs .venv39")
def test_keras_export_without_tensorflow_reports_converter_missing(tmp_path):
    (tmp_path / "m.keras").write_bytes(b"not really")
    res, _ = cli(PY39, ["--keras", str(tmp_path / "m.keras"), "--out", str(tmp_path / "out")], tmp_path)
    assert res["ok"] is False and res["error"]["code"] == "converter_missing"
    assert res["error"]["package"] == "tensorflow"


def test_bad_input_shape_is_reported(tmp_path):
    (tmp_path / "model.py").write_text("x = 1\n")
    res, _ = cli(sys.executable, ["--torch-file", str(tmp_path / "model.py"), "--class", "Net", "--input", "1,a", "--out", str(tmp_path / "out")], tmp_path)
    assert res["ok"] is False and res["error"]["code"] == "bad_input"


@pytest.mark.skipif(not os.path.exists(EXPORT_PY), reason="needs the .venv-export interpreter with litert-torch")
def test_python_api_export_torch_in_memory(tmp_path):
    code = textwrap.dedent(f"""
        import json, torch, torch.nn as nn
        import deepfriend
        m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        r = deepfriend.export(m, torch.randn(1, 4), out_dir={str(tmp_path / 'exp')!r}, name="tiny")
        print("RESULT", json.dumps(r))
    """)
    env = dict(os.environ, PYTHONPATH=PKG)
    p = subprocess.run([EXPORT_PY, "-c", code], capture_output=True, text=True, timeout=600, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")][-1]
    r = json.loads(line[7:])
    assert r["ok"] and r["outputShape"] == [1, 3] and r["verify"]["passed"], (r, p.stderr[-1500:])


def test_python_api_export_keras_in_memory(tmp_path):
    pytest.importorskip("tensorflow")
    code = textwrap.dedent(f"""
        import os, json
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import keras
        from keras import layers
        import deepfriend
        m = keras.Sequential([keras.Input(shape=(6,)), layers.Dense(2)])
        r = deepfriend.export(m, out_dir={str(tmp_path / 'k')!r})
        print("RESULT", json.dumps(r))
    """)
    env = dict(os.environ, PYTHONPATH=PKG, TF_CPP_MIN_LOG_LEVEL="3")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")][-1]
    r = json.loads(line[7:])
    assert r["ok"] and r["inputShape"] == [1, 6] and r["outputShape"] == [1, 2], r
