"""Keep a few tiny input/label samples from the first batch (for the "check data & labels" view)."""

import json
import math
import os
from typing import Any, Dict, List

MAX_ITEMS = 8
MAX_SIDE = 32


def _to_list(x: Any) -> Any:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        if hasattr(x, "float"):
            x = x.float()
        return x.tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _shape(v: Any) -> List[int]:
    shape = []
    while isinstance(v, list):
        shape.append(len(v))
        v = v[0] if v else None
    return shape


def _normalize(vals: List[float]) -> List[int]:
    finite = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return [0 for _ in vals]
    lo, hi = min(finite), max(finite)
    span = hi - lo or 1.0
    return [int(round((v - lo) / span * 255)) if isinstance(v, (int, float)) and math.isfinite(v) else 0 for v in vals]


def _image(item: Any, shape: List[int]) -> Dict[str, Any]:
    # (C, H, W) with C in 1/3, or (H, W), or (H, W, C) with C in 1/3
    if len(shape) == 3 and shape[0] in (1, 3):
        c, h, w = shape
        get = lambda ch, y, x: item[ch][y][x]
    elif len(shape) == 3 and shape[2] in (1, 3):
        h, w, c = shape
        get = lambda ch, y, x: item[y][x][ch]
    else:
        h, w = shape[-2], shape[-1]
        c = 1
        base = item
        while len(_shape(base)) > 2:
            base = base[0]
        get = lambda ch, y, x: base[y][x]
    sy, sx = max(1, math.ceil(h / MAX_SIDE)), max(1, math.ceil(w / MAX_SIDE))
    ys, xs = list(range(0, h, sy)), list(range(0, w, sx))
    raw = [float(get(ch, y, x)) for y in ys for x in xs for ch in range(c)]
    return {"shape": [len(ys), len(xs), c], "pixels": _normalize(raw)}


def _vector(item: Any) -> Dict[str, Any]:
    flat: List[float] = []

    def walk(v: Any) -> None:
        if isinstance(v, list):
            for e in v:
                if len(flat) >= 64:
                    return
                walk(e)
        elif isinstance(v, (int, float)):
            flat.append(float(v))

    walk(item)
    return {"shape": [len(flat)], "pixels": _normalize(flat)}


def _split(batch: Any) -> Any:
    if isinstance(batch, dict):
        keys = list(batch.keys())
        x = batch.get("image", batch.get("input", batch.get("x", batch[keys[0]] if keys else None)))
        y = batch.get("label", batch.get("labels", batch.get("y", batch.get("target"))))
        return x, y
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    return batch, None


def capture(run: Any, batch: Any) -> None:
    if not getattr(run, "ok", False):
        return
    try:
        x, y = _split(batch)
        xs = _to_list(x)
        ys = _to_list(y) if y is not None else None
        if not isinstance(xs, list):
            return
        items = []
        for i, item in enumerate(xs[:MAX_ITEMS]):
            shape = _shape(item)
            entry = _image(item, shape) if len(shape) >= 2 and shape[-1] >= 4 and shape[-2] >= 4 else _vector(item)
            label = ys[i] if isinstance(ys, list) and i < len(ys) else None
            if isinstance(label, list):
                label = max(range(len(label)), key=lambda k: label[k]) if label and all(isinstance(v, (int, float)) for v in label) else None
            if isinstance(label, float) and label.is_integer():
                label = int(label)
            entry["label"] = label
            items.append(entry)
        with open(os.path.join(run.dir, "samples.json"), "w", encoding="utf-8") as f:
            json.dump({"items": items}, f)
    except Exception:
        pass
