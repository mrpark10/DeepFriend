"""Run the same input through the original model and the .tflite file and compare."""

from typing import Any, Dict


def _interpreter(path: str) -> Any:
    try:
        from ai_edge_litert.interpreter import Interpreter  # LiteRT (preferred)
    except Exception:
        import tensorflow as tf  # deprecated fallback in TF 2.20+

        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    return it


def run_tflite(path: str, x: Any) -> Any:
    it = _interpreter(path)
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], x.astype(inp["dtype"]))
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


def compare(reference: Any, tflite_path: str, x: Any) -> Dict[str, Any]:
    import numpy as np

    y = np.asarray(run_tflite(tflite_path, x), dtype="float64").ravel()
    r = np.asarray(reference, dtype="float64").ravel()
    if y.shape != r.shape:
        return {"maxAbsDiff": None, "cosine": None, "passed": False}
    diff = float(np.max(np.abs(y - r))) if y.size else 0.0
    denom = float(np.linalg.norm(y) * np.linalg.norm(r))
    cosine = float(np.dot(y, r) / denom) if denom else 1.0
    scale = max(1.0, float(np.max(np.abs(r))) if r.size else 1.0)
    return {"maxAbsDiff": diff, "cosine": cosine, "passed": bool(cosine >= 0.999 and diff <= 1e-3 * scale)}
