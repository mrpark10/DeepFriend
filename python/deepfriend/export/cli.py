"""``python -m deepfriend.export.cli``: used by the "DeepFriend: Export to App" command.

The last stdout line is ``DEEPFRIEND_RESULT <json>``.
"""

import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, List, Optional

from .convert import ExportError, keras_to_tflite, torch_to_tflite
from .flutter import generate, notebook, safe_name


def progress(step: str) -> None:
    print("DF_PROGRESS " + step, flush=True)


def _parse_shape(text: str) -> List[int]:
    try:
        dims = [int(x) for x in text.replace("(", "").replace(")", "").split(",") if x.strip()]
    except ValueError:
        raise ExportError("bad_input", "input shape must be numbers like 1,3,224,224") from None
    if not dims or any(d < 1 for d in dims):
        raise ExportError("bad_input", "input shape must be positive numbers")
    return dims


def _write_flutter(out_dir: str, name: str, inp: List[int], outp: List[int], lang: str, layout: str) -> str:
    fdir = os.path.join(out_dir, "flutter")
    for rel, text in generate(name, inp, outp, lang=lang, layout=layout).items():
        path = os.path.join(fdir, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return fdir


def export_keras(path: str, out_dir: str, lang: str) -> Dict[str, Any]:
    progress("load")
    try:
        import numpy as np
        import keras
    except Exception:
        raise ExportError("converter_missing", "tensorflow/keras is not installed", package="tensorflow") from None
    try:
        model = keras.models.load_model(path)
    except Exception as e:
        raise ExportError("load_failed", "%s: %s" % (type(e).__name__, e)) from None
    inp = [1] + [int(d) for d in model.input_shape[1:]]
    name = safe_name(os.path.splitext(os.path.basename(path))[0])
    os.makedirs(out_dir, exist_ok=True)
    tfl = os.path.join(out_dir, name + ".tflite")
    progress("convert")
    keras_to_tflite(model, tfl)
    progress("verify")
    from .verify import compare

    x = np.random.RandomState(0).rand(*inp).astype("float32")
    ref = model.predict(x, verbose=0)
    ver = compare(ref, tfl, x)
    outp = [1] + [int(d) for d in np.asarray(ref).shape[1:]]
    progress("flutter")
    fdir = _write_flutter(out_dir, name, inp, outp, lang, "NHWC" if len(inp) == 4 else "NC")
    return {"ok": True, "tflite": tfl, "flutterDir": fdir, "inputShape": inp, "outputShape": outp, "verify": ver}


def _load_torch_class(model_file: str, class_name: str) -> Any:
    from ..shapes.extract import safe_source

    with open(model_file, encoding="utf-8") as f:
        text = f.read()
    tree = ast.parse(text)
    code = compile(safe_source(tree), model_file, "exec")
    sys.path.insert(0, os.path.dirname(os.path.abspath(model_file)))
    ns: Dict[str, Any] = {"__name__": "deepfriend_export"}
    exec(code, ns)
    cls = ns.get(class_name)
    if cls is None:
        raise ExportError("load_failed", "class %s not found in %s" % (class_name, os.path.basename(model_file)))
    return cls


def export_torch(model_file: str, class_name: str, weights: Optional[str], shape: List[int], out_dir: str, lang: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    name = safe_name(class_name)
    colab = os.path.join(out_dir, "convert_in_colab.ipynb")
    try:
        if sys.version_info < (3, 10):
            raise ExportError("platform_unsupported", "litert-torch needs Python 3.10+")
        progress("load")
        try:
            import torch
        except Exception:
            raise ExportError("converter_missing", "torch is not installed", package="torch") from None
        cls = _load_torch_class(model_file, class_name)
        try:
            model = cls()
        except TypeError as e:
            raise ExportError("ctor_args", str(e)) from None
        if weights:
            try:
                state = torch.load(weights, map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                    state = state["state_dict"]
                model.load_state_dict(state)
            except Exception as e:
                raise ExportError("load_failed", "%s: %s" % (type(e).__name__, e)) from None
        model.eval()
        example = torch.randn(*shape)
        tfl = os.path.join(out_dir, name + ".tflite")
        progress("convert")
        torch_to_tflite(model, example, tfl)
        progress("verify")
        from .verify import compare

        with torch.no_grad():
            ref = model(example).detach().cpu().numpy()
        ver = compare(ref, tfl, example.numpy())
        outp = list(ref.shape)
        progress("flutter")
        fdir = _write_flutter(out_dir, name, shape, outp, lang, "NCHW" if len(shape) == 4 else "NC")
        return {"ok": True, "tflite": tfl, "flutterDir": fdir, "inputShape": shape, "outputShape": outp, "verify": ver}
    except ExportError as e:
        if e.code == "platform_unsupported":
            with open(colab, "w", encoding="utf-8") as f:
                f.write(notebook(class_name, os.path.basename(model_file), os.path.basename(weights) if weights else "", shape, lang=lang))
            e.extra["colab"] = colab
        raise


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="deepfriend.export.cli")
    ap.add_argument("--keras")
    ap.add_argument("--torch-file")
    ap.add_argument("--class", dest="cls")
    ap.add_argument("--weights")
    ap.add_argument("--input")
    ap.add_argument("--out", default="deepfriend_export")
    ap.add_argument("--lang", default="ko")
    args = ap.parse_args(argv)
    result: Dict[str, Any]
    try:
        if args.keras:
            result = export_keras(args.keras, args.out, args.lang)
        elif args.torch_file and args.cls:
            shape = _parse_shape(args.input or "")
            result = export_torch(args.torch_file, args.cls, args.weights, shape, args.out, args.lang)
        else:
            raise ExportError("bad_input", "use --keras PATH or --torch-file FILE --class NAME --input SHAPE")
    except ExportError as e:
        result = {"ok": False, "error": {"code": e.code, "message": e.message, **{k: v for k, v in e.extra.items() if k != "colab"}}}
        if "colab" in e.extra:
            result["colab"] = e.extra["colab"]
    except Exception as e:  # never die without a result line
        result = {"ok": False, "error": {"code": "convert_failed", "message": "%s: %s" % (type(e).__name__, e)}}
    print("DEEPFRIEND_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
