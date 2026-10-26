"""In-memory export used by ``deepfriend.export()`` at the end of a training script."""

import os
import sys
from typing import Any, Dict, Optional

from .convert import ExportError, keras_to_tflite, torch_to_tflite
from .flutter import generate, safe_name


def _write(out_dir: str, name: str, inp: list, outp: list, lang: str, layout: str) -> str:
    fdir = os.path.join(out_dir, "flutter")
    for rel, text in generate(name, inp, outp, lang=lang, layout=layout).items():
        path = os.path.join(fdir, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return fdir


def export_model(model: Any, example: Any = None, out_dir: str = "deepfriend_export", name: Optional[str] = None, lang: str = "ko") -> Dict[str, Any]:
    from .verify import compare

    os.makedirs(out_dir, exist_ok=True)
    try:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(model, torch.nn.Module):
            if example is None:
                raise ExportError("bad_input", "PyTorch export needs an example input, e.g. torch.randn(1, 3, 224, 224)")
            n = safe_name(name or type(model).__name__)
            tfl = os.path.join(out_dir, n + ".tflite")
            model = model.cpu().eval()
            example = example.cpu()
            torch_to_tflite(model, example, tfl)
            with torch.no_grad():
                ref = model(example).detach().numpy()
            inp, outp = list(example.shape), list(ref.shape)
            ver = compare(ref, tfl, example.numpy())
            layout = "NCHW" if len(inp) == 4 else "NC"
        else:
            import numpy as np

            n = safe_name(name or getattr(model, "name", None) or "model")
            tfl = os.path.join(out_dir, n + ".tflite")
            keras_to_tflite(model, tfl)
            inp = [1] + [int(d) for d in model.input_shape[1:]]
            x = np.asarray(example, dtype="float32") if example is not None else np.random.RandomState(0).rand(*inp).astype("float32")
            ref = model.predict(x, verbose=0)
            outp = [1] + [int(d) for d in np.asarray(ref).shape[1:]]
            ver = compare(ref, tfl, x)
            layout = "NHWC" if len(inp) == 4 else "NC"
        fdir = _write(out_dir, n, inp, outp, lang, layout)
        res = {"ok": True, "tflite": tfl, "flutterDir": fdir, "inputShape": inp, "outputShape": outp, "verify": ver}
        print("[deepfriend] exported %s (verify passed: %s)" % (tfl, ver["passed"]))
        return res
    except ExportError as e:
        print("[deepfriend] export failed: %s" % e.message)
        return {"ok": False, "error": {"code": e.code, "message": e.message, **e.extra}}
