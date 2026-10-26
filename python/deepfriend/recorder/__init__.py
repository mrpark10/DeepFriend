"""Training-side recorder behind ``deepfriend.watch()``."""

import sys
from typing import Any, Dict, Optional

from .run import Run

_current: Optional[Run] = None


class _NullRun:
    """Returned when recording could not start; every method is a no-op."""

    ok = False
    id = None
    dir = None

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


def watch(name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, record: Optional[bool] = None) -> Any:
    global _current
    if _current is not None:
        return _current
    try:
        run = Run(name=name, config=config, record=record)
    except Exception:
        return _NullRun()
    _current = run
    try:
        from . import excepthook

        excepthook.install(run)
    except Exception:
        pass
    try:
        from . import snapshot

        snapshot.capture(run)
    except Exception:
        pass
    if "torch" in sys.modules:
        try:
            from . import torch_hooks

            torch_hooks.install(run)
        except Exception:
            pass
    if any(m in sys.modules for m in ("keras", "tensorflow", "tf_keras")):
        try:
            from . import keras_cb

            keras_cb.patch_fit(run)
        except Exception:
            pass
    return run


def log(step: Optional[int] = None, epoch: Optional[int] = None, **metrics: Any) -> None:
    run = _current if _current is not None else watch()
    try:
        run.log(step=step, epoch=epoch, **metrics)
    except Exception:
        pass


def stop(status: str = "finished") -> None:
    global _current
    run = _current
    if run is None:
        return
    try:
        from . import torch_hooks

        torch_hooks.uninstall(run)
    except Exception:
        pass
    run.stop(status)
    _current = None


def keras_callback() -> Any:
    from . import keras_cb

    run = _current if _current is not None else watch()
    return keras_cb.make_callback(run)
