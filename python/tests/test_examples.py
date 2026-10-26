"""The bundled examples must analyze cleanly and actually run (offline, CPU)."""
import os
import shutil
import subprocess
import sys

import pytest

from deepfriend.shapes.analyze import Analyzer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EX = os.path.join(ROOT, "examples")
PKG = os.path.join(ROOT, "python")
CLEAN = ["01_mnist_mlp.py", "02_cifar_cnn.py", "03_keras_cnn.py"]


def analyze(name):
    with open(os.path.join(EX, name), encoding="utf-8") as f:
        text = f.read()
    return Analyzer().analyze(name, text, runtime=False)


@pytest.mark.parametrize("name", CLEAN)
def test_clean_examples_have_hints_and_no_errors(name):
    res = analyze(name)
    assert res["syntaxError"] is None
    models = [m for m in res["models"] if not m["needsInput"]]
    assert models, "every example model has an input shape"
    for m in res["models"]:
        assert m["diagnostics"] == [], (name, m["name"], m["diagnostics"])
        if not m["needsInput"]:
            assert m["hints"] and all(h["shape"] != "?" for h in m["hints"]), (name, m["hints"])


def test_shape_error_example_has_exactly_one_mismatch_and_a_codelens_model():
    res = analyze("04_shape_error.py")
    diags = [d for m in res["models"] for d in m["diagnostics"]]
    assert {d["code"] for d in diags} == {"linear_in"}
    assert len({(d["line"]) for d in diags}) == 2  # definition + forward call
    assert any(m["needsInput"] for m in res["models"])


def run(name, tmp_path):
    work = tmp_path / "ex"
    shutil.copytree(EX, work)
    env = dict(os.environ, PYTHONPATH=PKG, TF_CPP_MIN_LOG_LEVEL="3", PYTHONIOENCODING="utf-8")
    env.pop("DEEPFRIEND_WORKSPACE", None)
    return subprocess.run([sys.executable, name], cwd=str(work), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300), work


@pytest.mark.parametrize("name", ["01_mnist_mlp.py", "02_cifar_cnn.py"])
def test_torch_examples_run_and_record(name, tmp_path):
    pytest.importorskip("torch")
    p, work = run(name, tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    runs = os.listdir(work / ".deepfriend" / "runs")
    assert len(runs) == 1


def test_keras_example_runs(tmp_path):
    pytest.importorskip("tensorflow")
    p, work = run("03_keras_cnn.py", tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert len(os.listdir(work / ".deepfriend" / "runs")) == 1


def test_shape_error_example_fails_with_the_matmul_error(tmp_path):
    pytest.importorskip("torch")
    p, _ = run("04_shape_error.py", tmp_path)
    assert p.returncode != 0
    assert "mat1 and mat2 shapes cannot be multiplied" in p.stderr
