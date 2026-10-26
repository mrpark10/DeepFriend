"""Model → .tflite converters.

Checked 2026-10-26: PyTorch uses ``litert-torch`` (formerly ai-edge-torch, 0.10.0,
officially Linux; verified working on macOS arm64; Python >= 3.10, torch >= 2.4).
Keras uses TensorFlow's built-in TFLiteConverter.
"""

import sys
from typing import Any


class ExportError(Exception):
    def __init__(self, code: str, message: str = "", **extra: Any) -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.extra = extra


def keras_to_tflite(model: Any, out_path: str) -> str:
    try:
        import tensorflow as tf
    except Exception:
        raise ExportError("converter_missing", "tensorflow is not installed", package="tensorflow") from None
    try:
        data = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    except Exception as e:
        raise ExportError("convert_failed", "%s: %s" % (type(e).__name__, e)) from None
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def torch_to_tflite(model: Any, example: Any, out_path: str) -> str:
    if sys.version_info < (3, 10):
        raise ExportError("platform_unsupported", "litert-torch needs Python 3.10+")
    try:
        import litert_torch
    except Exception:
        raise ExportError("converter_missing", "litert-torch is not installed", package="litert-torch") from None
    try:
        edge = litert_torch.convert(model.eval(), (example,))
        edge.export(out_path)
    except Exception as e:
        if sys.platform == "win32":
            raise ExportError("platform_unsupported", "%s: %s" % (type(e).__name__, e)) from None
        raise ExportError("convert_failed", "%s: %s" % (type(e).__name__, e)) from None
    return out_path
