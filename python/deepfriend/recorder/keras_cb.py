"""Keras instrumentation: a Callback, plus a Model.fit patch that adds it automatically."""

import sys
from typing import Any, Dict, Optional


def _keras() -> Any:
    for name in ("keras", "tf_keras"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "callbacks"):
            return mod
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        return tf.keras
    import keras  # noqa: F401  (explicit keras_callback() call)

    return keras


def _float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        try:
            import numpy as np

            return float(np.asarray(v))
        except Exception:
            return None


def _acc(logs: Dict[str, Any], prefix: str = "") -> Optional[float]:
    for k in ("accuracy", "acc", "sparse_categorical_accuracy", "categorical_accuracy", "binary_accuracy"):
        if prefix + k in logs:
            return _float(logs[prefix + k])
    return None


def make_callback(run: Any) -> Any:
    keras = _keras()

    class DeepFriendCallback(keras.callbacks.Callback):  # type: ignore[misc,name-defined]
        _deepfriend = True

        def on_train_begin(self, logs: Any = None) -> None:
            try:
                run.set_framework("tensorflow")
                params = getattr(self, "params", None) or {}
                if "epochs" in params:
                    run.set_config("epochs", int(params["epochs"]))
                opt = getattr(self.model, "optimizer", None)
                if opt is not None:
                    run.set_config("optimizer", type(opt).__name__)
                    lr = getattr(opt, "learning_rate", None)
                    lr = _float(lr() if callable(lr) and not hasattr(lr, "numpy") else lr)
                    if lr is not None:
                        run.set_config("lr", lr)
                try:
                    run.meta["model"]["params"] = int(self.model.count_params())
                    run.meta["model"]["name"] = getattr(self.model, "name", None)
                except Exception:
                    pass
                run.write_meta()
            except Exception:
                pass

        def on_train_batch_end(self, batch: Any, logs: Any = None) -> None:
            logs = logs or {}
            if "loss" in logs:
                run.train_step(_float(logs["loss"]), _acc(logs))

        def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
            logs = logs or {}
            try:
                with run._lock:
                    # Keras already averages per epoch: replace our per-step running mean by its numbers
                    run._agg["train"] = {"loss": [], "acc": []}
                    if "loss" in logs:
                        run._agg["train"]["loss"].append(_float(logs["loss"]))
                    a = _acc(logs)
                    if a is not None:
                        run._agg["train"]["acc"].append(a)
                    if "val_loss" in logs:
                        run._agg["val"]["loss"].append(_float(logs["val_loss"]))
                    va = _acc(logs, "val_")
                    if va is not None:
                        run._agg["val"]["acc"].append(va)
                    if run.meta.get("numClasses") is None and (a is not None or va is not None):
                        try:
                            n = int(self.model.output_shape[-1])
                            run.meta["numClasses"] = 2 if n == 1 else n
                        except Exception:
                            pass
                    run._flush_agg()
                    run.epoch = epoch + 1
                    run._steps_in_epoch = 0
                    run._saw_val_since_train = False
            except Exception:
                pass

    return DeepFriendCallback()


def patch_fit(run: Any) -> None:
    """Add the callback to every Model.fit call that does not already have one."""
    targets = []
    for name in ("keras", "tf_keras"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "Model"):
            targets.append(mod.Model)
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        try:
            targets.append(tf.keras.Model)
        except Exception:
            pass
    seen = set()
    for cls in targets:
        if id(cls) in seen or getattr(cls.fit, "_deepfriend", False):
            continue
        seen.add(id(cls))
        orig = cls.fit

        def fit(self: Any, *args: Any, __orig: Any = orig, **kwargs: Any) -> Any:
            try:
                cbs = list(kwargs.get("callbacks") or [])
                if not any(getattr(c, "_deepfriend", False) for c in cbs):
                    cbs.append(make_callback(run))
                kwargs["callbacks"] = cbs
                bs = kwargs.get("batch_size")
                run.set_config("batch_size", int(bs) if bs else 32, overwrite=True)
                x = args[0] if args else kwargs.get("x")
                y = args[1] if len(args) > 1 else kwargs.get("y")
                if x is not None and not getattr(run, "_keras_sample", False):
                    run._keras_sample = True
                    from . import samples

                    samples.capture(run, (x[:8], y[:8] if y is not None else None))
            except Exception:
                pass
            return __orig(self, *args, **kwargs)

        fit._deepfriend = True  # type: ignore[attr-defined]
        cls.fit = fit
