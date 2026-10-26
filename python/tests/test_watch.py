"""deepfriend.watch(): runs real training scripts in a temporary project and checks the run folder."""
import json
import os
import stat
import subprocess
import sys
import textwrap

import pytest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(project, code, env_extra=None, timeout=240, expect_ok=True):
    project.mkdir(parents=True, exist_ok=True)
    script = project / "train.py"
    script.write_text(textwrap.dedent(code))
    env = dict(os.environ, PYTHONPATH=PKG_ROOT, PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="3")
    env.pop("DEEPFRIEND_WORKSPACE", None)
    env.pop("DEEPFRIEND_RECORD", None)
    env.update(env_extra or {})
    p = subprocess.run([sys.executable, str(script)], cwd=str(project), env=env, capture_output=True, text=True, timeout=timeout)
    if expect_ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def only_run(project):
    runs = project / ".deepfriend" / "runs"
    dirs = sorted(d for d in runs.iterdir() if d.is_dir())
    assert len(dirs) == 1, dirs
    return dirs[0]


def metrics(run_dir):
    out = []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def meta(run_dir):
    return json.loads((run_dir / "meta.json").read_text())


TORCH_TRAIN = """
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepfriend
deepfriend.watch(name="mlp", config={"note": "hello"})
torch.manual_seed(0)
X = torch.randn(96, 8); y = torch.randint(0, 3, (96,))
train = DataLoader(TensorDataset(X[:64], y[:64]), batch_size=16, shuffle=True)
val = DataLoader(TensorDataset(X[64:], y[64:]), batch_size=16)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss()
for epoch in range(2):
    model.train()
    for xb, yb in train:
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        for xb, yb in val:
            crit(model(xb), yb)
"""


def test_torch_training_is_recorded(tmp_path):
    pytest.importorskip("torch")
    project = tmp_path / "proj"
    run_script(project, TORCH_TRAIN)
    rd = only_run(project)
    m = meta(rd)
    assert m["status"] == "finished" and m["endedAt"]
    assert m["name"] == "mlp" and m["script"] == "train.py"
    assert m["framework"] == "pytorch"
    assert m["config"]["lr"] == 0.01 and m["config"]["batch_size"] == 16 and m["config"]["optimizer"] == "Adam"
    assert m["config"]["note"] == "hello"
    assert m["numClasses"] == 3
    assert m["model"]["params"] == 8 * 16 + 16 + 16 * 3 + 3
    assert m["deepfriendVersion"] == "0.1.0"
    assert m["record"] is True
    recs = metrics(rd)
    train_steps = [r for r in recs if r["split"] == "train" and not r.get("agg")]
    assert len(train_steps) == 8
    assert [r["step"] for r in train_steps] == list(range(1, 9))
    assert [r["epoch"] for r in train_steps] == [0] * 4 + [1] * 4
    assert all(0 <= r["acc"] <= 1 for r in train_steps)
    val_agg = [r for r in recs if r["split"] == "val" and r.get("agg")]
    train_agg = [r for r in recs if r["split"] == "train" and r.get("agg")]
    assert [r["epoch"] for r in val_agg] == [0, 1] and [r["epoch"] for r in train_agg] == [0, 1]
    assert all(isinstance(r["loss"], float) for r in val_agg)
    assert m["final"]["val_loss"] == val_agg[-1]["loss"]
    assert "val_loss" in m["best"]
    samples = json.loads((rd / "samples.json").read_text())
    assert 1 <= len(samples["items"]) <= 8 and samples["items"][0]["shape"] == [8]
    assert (rd / "code" / "train.py").read_text().startswith("\nimport torch")
    assert (project / ".deepfriend" / ".gitignore").read_text().strip() == "*"


def test_functional_loss_and_manual_loop_epochs(tmp_path):
    pytest.importorskip("torch")
    project = tmp_path / "p2"
    run_script(project, """
import torch, torch.nn as nn, torch.nn.functional as F
import deepfriend
deepfriend.watch()
torch.manual_seed(0)
model = nn.Linear(4, 5)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
X = torch.randn(32, 4); y = torch.randint(0, 5, (32,))
for epoch in range(3):
    for i in range(0, 32, 8):
        opt.zero_grad()
        loss = F.cross_entropy(model(X[i:i+8]), y[i:i+8])
        loss.backward(); opt.step()
    with torch.no_grad():
        F.cross_entropy(model(X), y)
""")
    rd = only_run(project)
    m = meta(rd)
    assert m["numClasses"] == 5 and m["config"]["optimizer"] == "SGD"
    recs = metrics(rd)
    assert [r["epoch"] for r in recs if r["split"] == "val" and r.get("agg")] == [0, 1, 2]


def test_scaled_loss_records_the_real_value(tmp_path):
    pytest.importorskip("torch")
    project = tmp_path / "p3"
    run_script(project, """
import torch, torch.nn as nn, torch.nn.functional as F
import deepfriend
deepfriend.watch()
model = nn.Linear(2, 2)
x = torch.ones(4, 2); y = torch.zeros(4, dtype=torch.long)
loss = F.cross_entropy(model(x), y)
real = loss.item()
(loss * 1024).backward()
open("real.txt", "w").write(repr(real))
""")
    real = float((project / "real.txt").read_text())
    step = [r for r in metrics(only_run(project)) if r["split"] == "train" and not r.get("agg")][0]
    assert abs(step["loss"] - real) < 1e-6


def test_keras_fit_is_recorded_without_explicit_callback(tmp_path):
    pytest.importorskip("tensorflow")
    project = tmp_path / "pk"
    run_script(project, """
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import keras
from keras import layers
import deepfriend
deepfriend.watch()
x = np.random.rand(64, 4).astype("float32"); y = np.random.randint(0, 3, 64)
m = keras.Sequential([keras.Input(shape=(4,)), layers.Dense(8, activation="relu"), layers.Dense(3, activation="softmax")])
m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
m.fit(x, y, validation_split=0.25, epochs=2, batch_size=16, verbose=0)
""")
    rd = only_run(project)
    m = meta(rd)
    assert m["framework"] == "tensorflow" and m["status"] == "finished"
    assert m["config"]["epochs"] == 2 and m["config"]["batch_size"] == 16
    assert abs(m["config"]["lr"] - 0.005) < 1e-9
    assert m["numClasses"] == 3
    recs = metrics(rd)
    assert [r["epoch"] for r in recs if r["split"] == "val" and r.get("agg")] == [0, 1]
    assert any("acc" in r for r in recs if r["split"] == "train")


def test_nan_loss_written_as_null_flag(tmp_path):
    pytest.importorskip("torch")
    project = tmp_path / "pn"
    run_script(project, """
import torch
import deepfriend
deepfriend.watch()
w = torch.ones(1, requires_grad=True)
(w * float("nan")).sum().backward()
""")
    step = [r for r in metrics(only_run(project)) if r["split"] == "train"][0]
    assert step["loss"] is None and step["nan"] is True


def test_exception_writes_error_json(tmp_path):
    project = tmp_path / "pe"
    p = run_script(project, """
import deepfriend
deepfriend.watch()
def f():
    raise ValueError("bad config")
f()
""", expect_ok=False)
    assert "ValueError: bad config" in p.stderr  # the original traceback is still printed
    rd = only_run(project)
    assert meta(rd)["status"] == "failed"
    err = json.loads((rd / "error.json").read_text())
    assert err["type"] == "ValueError" and err["message"] == "bad config"
    assert err["frames"][-1]["name"] == "f" and err["frames"][-1]["line"] == 5
    assert err["frames"][-1]["file"].endswith("train.py")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_keyboard_interrupt_marks_stopped(tmp_path):
    project = tmp_path / "pi"
    run_script(project, """
import os, signal, time
import deepfriend
deepfriend.watch()
os.kill(os.getpid(), signal.SIGINT)
time.sleep(5)
""", expect_ok=False)
    rd = only_run(project)
    assert meta(rd)["status"] == "stopped"
    assert not (rd / "error.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_watch_never_raises_in_read_only_project(tmp_path):
    project = tmp_path / "ro"
    project.mkdir()
    (project / "train.py").write_text("import deepfriend\nr = deepfriend.watch()\ndeepfriend.log(loss=1.0)\nprint('trained ok')\n")
    os.chmod(project, stat.S_IRUSR | stat.S_IXUSR)
    try:
        env = dict(os.environ, PYTHONPATH=PKG_ROOT)
        env.pop("DEEPFRIEND_WORKSPACE", None)
        p = subprocess.run([sys.executable, str(project / "train.py")], cwd=str(project), env=env, capture_output=True, text=True, timeout=60)
    finally:
        os.chmod(project, stat.S_IRWXU)
    assert p.returncode == 0, p.stderr
    assert "trained ok" in p.stdout


def test_record_false_cleans_up(tmp_path):
    project = tmp_path / "pr"
    run_script(project, """
import deepfriend
r = deepfriend.watch(record=False)
deepfriend.log(loss=0.5)
assert "live" in str(r.dir)
""")
    assert not (project / ".deepfriend" / "runs").exists() or not any((project / ".deepfriend" / "runs").iterdir())
    live = project / ".deepfriend" / "live"
    assert not live.exists() or not any(live.iterdir())


def test_env_disables_recording(tmp_path):
    project = tmp_path / "pv"
    run_script(project, "import deepfriend\nr = deepfriend.watch()\nassert 'live' in str(r.dir)\n", env_extra={"DEEPFRIEND_RECORD": "0"})


def test_workspace_env_sets_the_root(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    sub = ws / "src" / "deep"
    run_script(sub, "import deepfriend\ndeepfriend.watch()\n", env_extra={"DEEPFRIEND_WORKSPACE": str(ws)})
    rd = only_run(ws)
    assert meta(rd)["script"] == "src/deep/train.py"


def test_root_found_by_walking_up(tmp_path):
    proj = tmp_path / "walk"
    (proj / ".git").mkdir(parents=True)
    run_script(proj / "a" / "b", "import deepfriend\ndeepfriend.watch()\n")
    assert only_run(proj)


def test_watch_is_idempotent_and_log_works(tmp_path):
    project = tmp_path / "pl"
    run_script(project, """
import deepfriend
a = deepfriend.watch()
b = deepfriend.watch()
assert a is b
deepfriend.log(loss=2.0, acc=0.5)
deepfriend.log(loss=1.0, acc=0.75)
deepfriend.log(epoch=0, val_loss=1.5, val_acc=0.6)
""")
    rd = only_run(project)
    recs = metrics(rd)
    assert [r["loss"] for r in recs if r["split"] == "train" and not r.get("agg")] == [2.0, 1.0]
    v = [r for r in recs if r["split"] == "val"]
    assert v[0]["loss"] == 1.5 and v[0]["acc"] == 0.6


def test_stdlib_only_import_and_watch(tmp_path):
    """Must work on a Python without torch/tensorflow (run by .venv39 too)."""
    project = tmp_path / "plain"
    run_script(project, "import sys\nimport deepfriend\ndeepfriend.watch()\nassert 'torch' not in sys.modules\n")
    assert meta(only_run(project))["status"] == "finished"


def test_overhead_is_small(tmp_path):
    pytest.importorskip("torch")
    body = """
import time, torch, torch.nn as nn, torch.nn.functional as F
WATCH
torch.manual_seed(0)
m = nn.Linear(16, 4); opt = torch.optim.SGD(m.parameters(), lr=0.01)
x = torch.randn(32, 16); y = torch.randint(0, 4, (32,))
for _ in range(50):
    opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
t = time.perf_counter()
for _ in range(2000):
    opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
open("t.txt", "w").write(repr(time.perf_counter() - t))
"""
    base = tmp_path / "base"
    run_script(base, body.replace("WATCH", ""))
    watched = tmp_path / "watched"
    run_script(watched, body.replace("WATCH", "import deepfriend; deepfriend.watch()"))
    t0 = float((base / "t.txt").read_text())
    t1 = float((watched / "t.txt").read_text())
    assert t1 <= max(t0 * 1.6, t0 + 0.15), (t0, t1)


def test_star_set_by_the_extension_survives_meta_rewrites(tmp_path):
    project = tmp_path / "star"
    run_script(project, """
import json, os
import deepfriend
r = deepfriend.watch()
path = os.path.join(r.dir, "meta.json")
m = json.load(open(path)); m["starred"] = True
json.dump(m, open(path, "w"))
deepfriend.log(loss=1.0)
deepfriend.log(epoch=1, val_loss=0.5)
""")
    assert meta(only_run(project))["starred"] is True
