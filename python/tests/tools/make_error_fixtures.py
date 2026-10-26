"""Capture real Python/PyTorch/TensorFlow tracebacks as F3 test fixtures.

Run with the dev venv:  .venv/bin/python python/tests/tools/make_error_fixtures.py
Writes extension/src/test/fixtures/errors/<id>.txt (stderr of each failing script).
Errors that cannot happen on this machine (CUDA OOM, ...) live in hand-written
fixtures named *.synthetic.txt and are not touched here.
"""
import os
import subprocess
import sys
import tempfile

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "extension", "src", "test", "fixtures", "errors")

T = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
K = "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL']='3'\nimport numpy as np\nimport keras\nfrom keras import layers\n"

SCRIPTS = {
    "shape_matmul": T + "fc = nn.Linear(10000, 10)\nx = torch.zeros(32, 12544)\nfc(x)\n",
    "shape_conv_channels": T + "c = nn.Conv2d(3, 16, 3)\nc(torch.zeros(1, 1, 28, 28))\n",
    "shape_tensor_size": T + "a = torch.zeros(2, 3)\nb = torch.zeros(2, 4)\na + b\n",
    "shape_view": T + "x = torch.zeros(2, 3, 4)\nx.view(5, -1)\n",
    "shape_target_batch": T + "F.cross_entropy(torch.zeros(32, 10), torch.zeros(64, dtype=torch.long))\n",
    "shape_mse_warning_free": T + "F.mse_loss(torch.zeros(4, 3), torch.zeros(4, 5))\n",
    "dtype_matmul": T + "fc = nn.Linear(4, 2)\nfc(torch.zeros(3, 4, dtype=torch.float64))\n",
    "dtype_long_target": T + "F.cross_entropy(torch.zeros(4, 3), torch.zeros(4, dtype=torch.int32))\n",
    "dtype_float_target": T + "F.nll_loss(torch.zeros(4, 3), torch.zeros(4))\n",
    "cuda_unavailable": T + "torch.zeros(3).cuda()\n",
    "module_not_found": "import cv2\n",
    "indentation": "def f():\nreturn 1\n",
    "syntax_error": "x = (1,\ny = 2\n",
    "name_error": "print(modle)\n",
    "attribute_none": "x = None\nx.shape\n",
    "attribute_error": T + "m = nn.Linear(2, 2)\nm.fc1\n",
    "type_args": T + "class M(nn.Module):\n    def forward(self, x):\n        return x\nM()(1, 2)\n",
    "index_error": "a = [1, 2]\na[5]\n",
    "key_error": "d = {'lr': 1}\nd['batch_size']\n",
    "file_not_found": "open('data/train.csv')\n",
    "zero_division": "1 / 0\n",
    "chained": "try:\n    {}['x']\nexcept KeyError:\n    raise ValueError('bad config')\n",
    "keras_input": K + "m = keras.Sequential([keras.Input(shape=(784,)), layers.Dense(10)])\nm.predict(np.zeros((2, 28, 28)), verbose=0)\n",
    "keras_target": K + "m = keras.Sequential([keras.Input(shape=(4,)), layers.Dense(10, activation='softmax')])\nm.compile(loss='categorical_crossentropy')\nm.fit(np.zeros((8, 4)), np.zeros((8, 1)), verbose=0)\n",
}

MPS = {
    "device_mismatch": T + "a = torch.zeros(2, device='mps')\nb = torch.zeros(2)\ntorch.matmul(a, b)\n",
    "device_input_weight": T + "c = nn.Linear(2, 2)\nc(torch.zeros(1, 2, device='mps'))\n",
}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    scripts = dict(SCRIPTS)
    try:
        import torch

        if torch.backends.mps.is_available():
            scripts.update(MPS)
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as d:
        for name, code in scripts.items():
            path = os.path.join(d, "train.py")
            with open(path, "w") as f:
                f.write(code)
            p = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=d, timeout=300)
            if p.returncode == 0:
                print("NO ERROR:", name)
                continue
            text = p.stderr.replace(d, "/home/student/project").replace(sys.prefix, "/home/student/.venv")
            lines = [l for l in text.splitlines() if "UserWarning" not in l and "warnings.warn" not in l and "absl" not in l and not l.startswith("WARNING")]
            with open(os.path.join(OUT, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            print("ok", name, "->", lines[-1][:100] if lines else "")


if __name__ == "__main__":
    main()
