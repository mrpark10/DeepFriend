"""PyTorch instrumentation for watch().

* functional losses (used by both ``nn.*Loss`` modules and ``F.*``) report the
  real loss value, accuracy for classification losses, and validation losses
  (computed while gradients are disabled);
* ``Tensor.backward`` marks one training step (falls back to the backward
  tensor's value when no loss function was seen, e.g. custom losses);
* a global optimizer pre-step hook records the optimizer name and learning rate;
* ``DataLoader.__iter__`` gives the batch size, epoch boundaries and a sample batch.

Everything is wrapped in try/except: instrumentation must never break training.
"""

import threading
from typing import Any, Dict, List, Optional, Tuple

_LOSSES = {
    "cross_entropy": "ce",
    "nll_loss": "ce",
    "binary_cross_entropy": "bce",
    "binary_cross_entropy_with_logits": "bce_logits",
    "mse_loss": "reg",
    "l1_loss": "reg",
    "smooth_l1_loss": "reg",
    "huber_loss": "reg",
    "kl_div": "reg",
}

_tls = threading.local()


class _State:
    def __init__(self) -> None:
        self.pending: Optional[Tuple[float, Optional[float]]] = None  # (loss, acc) from the last training loss call
        self.loader_iters: Dict[int, int] = {}
        self.active_loader: Optional[int] = None
        self.train_loader: Optional[int] = None
        self.sample_taken = False
        self.params_counted = False
        self.restore: List[Tuple[Any, str, Any]] = []
        self.handles: List[Any] = []


def _accuracy(torch: Any, kind: str, inp: Any, target: Any) -> Tuple[Optional[float], Optional[int]]:
    try:
        with torch.no_grad():
            if kind == "ce":
                if inp.dim() < 2:
                    return None, None
                classes = int(inp.shape[1])
                pred = inp.argmax(1)
                tgt = target.argmax(1) if target.is_floating_point() and target.dim() == inp.dim() else target
                if tgt.shape != pred.shape:
                    return None, classes
                valid = tgt != -100
                n = int(valid.sum())
                if n == 0:
                    return None, classes
                return float((pred[valid] == tgt[valid]).float().sum()) / n, classes
            if kind in ("bce", "bce_logits"):
                thr = 0.0 if kind == "bce_logits" else 0.5
                if inp.shape != target.shape:
                    return None, 2
                return float(((inp > thr) == (target > 0.5)).float().mean()), 2
    except Exception:
        pass
    return None, None


def install(run: Any) -> None:
    import torch
    import torch.nn.functional as F

    state = _State()
    run._torch_state = state
    run.set_framework("pytorch")

    def wrap_loss(name: str, kind: str) -> None:
        orig = getattr(F, name, None)
        if orig is None:
            return

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            out = orig(*args, **kwargs)
            if getattr(_tls, "busy", False):
                return out
            _tls.busy = True
            try:
                if torch.is_tensor(out) and out.numel() == 1:
                    value = float(out.detach())
                    acc, classes = (None, None)
                    if kind != "reg" and len(args) >= 2 and torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
                        acc, classes = _accuracy(torch, kind, args[0], args[1])
                    if classes and run.meta.get("numClasses") is None:
                        run.meta["numClasses"] = classes
                        run.write_meta()
                    if torch.is_grad_enabled() and out.requires_grad:
                        state.pending = (value, acc)
                    else:
                        run.val_value(value, acc)
            except Exception:
                pass
            finally:
                _tls.busy = False
            return out

        wrapper.__wrapped__ = orig  # type: ignore[attr-defined]
        setattr(F, name, wrapper)
        state.restore.append((F, name, orig))

    for name, kind in _LOSSES.items():
        wrap_loss(name, kind)

    orig_backward = torch.Tensor.backward

    def backward(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            if self.numel() == 1:
                if state.train_loader is None and state.active_loader is not None:
                    state.train_loader = state.active_loader
                if state.pending is not None:
                    value, acc = state.pending
                else:
                    value, acc = float(self.detach()), None
                state.pending = None
                run.train_step(value, acc, run._last_lr_seen if hasattr(run, "_last_lr_seen") else None)
        except Exception:
            pass
        return orig_backward(self, *args, **kwargs)

    torch.Tensor.backward = backward
    state.restore.append((torch.Tensor, "backward", orig_backward))

    try:
        from torch.optim.optimizer import register_optimizer_step_pre_hook

        def pre_step(opt: Any, _args: Any, _kwargs: Any) -> None:
            try:
                lr = opt.param_groups[0].get("lr")
                lr = float(lr)
                run._last_lr_seen = lr
                run.set_config("optimizer", type(opt).__name__)
                run.set_config("lr", lr)
                if not state.params_counted:
                    state.params_counted = True
                    total = sum(p.numel() for g in opt.param_groups for p in g["params"])
                    run.meta["model"]["params"] = int(total)
                    run.write_meta()
            except Exception:
                pass

        state.handles.append(register_optimizer_step_pre_hook(pre_step))
    except Exception:
        pass

    try:
        from torch.utils.data import DataLoader

        orig_iter = DataLoader.__iter__

        def loader_iter(self: Any) -> Any:
            it = orig_iter(self)
            try:
                key = id(self)
                state.loader_iters[key] = state.loader_iters.get(key, 0) + 1
                state.active_loader = key
                if key == state.train_loader and state.loader_iters[key] > 1:
                    run.new_epoch_from_loader()
                if self.batch_size is not None and (state.train_loader is None or key == state.train_loader):
                    run.set_config("batch_size", int(self.batch_size), overwrite=state.train_loader == key)
                if not state.sample_taken:
                    return _SampleIter(it, run, state)
            except Exception:
                pass
            return it

        DataLoader.__iter__ = loader_iter  # type: ignore[assignment]
        state.restore.append((DataLoader, "__iter__", orig_iter))
    except Exception:
        pass


class _SampleIter:
    """Transparent iterator that keeps the first batch as samples."""

    def __init__(self, it: Any, run: Any, state: _State) -> None:
        self._it = it
        self._run = run
        self._state = state

    def __iter__(self) -> "_SampleIter":
        return self

    def __next__(self) -> Any:
        batch = next(self._it)
        if not self._state.sample_taken:
            self._state.sample_taken = True
            try:
                from . import samples

                samples.capture(self._run, batch)
            except Exception:
                pass
        return batch

    def __len__(self) -> int:
        return len(self._it)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._it, name)


def uninstall(run: Any) -> None:
    state: Optional[_State] = getattr(run, "_torch_state", None)
    if state is None:
        return
    for obj, name, orig in reversed(state.restore):
        try:
            setattr(obj, name, orig)
        except Exception:
            pass
    for h in state.handles:
        try:
            h.remove()
        except Exception:
            pass
