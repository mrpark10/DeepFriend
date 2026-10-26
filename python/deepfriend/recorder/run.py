"""One training run: the ``.deepfriend/runs/<id>/`` folder and everything written into it.

Every public method swallows its own errors: recording must never break training.
"""

import atexit
import datetime
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from .. import __version__
from ..procutil import no_window_kwargs

FLUSH_SECONDS = 0.5


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    return f


def find_root(script: Optional[str]) -> str:
    """Project root: $DEEPFRIEND_WORKSPACE, else walk up looking for .deepfriend/.git/.vscode, else cwd."""
    ws = os.environ.get("DEEPFRIEND_WORKSPACE")
    if ws and os.path.isdir(ws):
        return os.path.abspath(ws)
    start = os.path.dirname(os.path.abspath(script)) if script else os.getcwd()
    cur = start
    while True:
        for marker in (".deepfriend", ".git", ".vscode"):
            if os.path.exists(os.path.join(cur, marker)):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.getcwd() if not script else start


def main_script() -> Optional[str]:
    main = sys.modules.get("__main__")
    f = getattr(main, "__file__", None)
    if f and os.path.isfile(f):
        return os.path.abspath(f)
    return None


def _git_hash(root: str) -> Optional[str]:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=1, **no_window_kwargs())
        if p.returncode == 0:
            return p.stdout.strip() or None
    except Exception:
        pass
    return None


def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader holds the file for a moment
            time.sleep(0.05)
    try:
        os.remove(tmp)
    except OSError:
        pass


class Run:
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, record: Optional[bool] = None) -> None:
        if record is None:
            record = os.environ.get("DEEPFRIEND_RECORD", "1") != "0"
        self.record = record
        self.script = main_script()
        self.root = find_root(self.script)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.id = "%s-%04x" % (stamp, random.randrange(0x10000))
        base = os.path.join(self.root, ".deepfriend")
        self.dir = os.path.join(base, "runs" if record else "live", self.id)
        self.ok = False
        self._lock = threading.RLock()
        self._pending: List[str] = []
        self._stop_event = threading.Event()
        self.status = "running"
        self.step = 0
        self.epoch = 0
        self._steps_in_epoch = 0
        self._saw_val_since_train = False
        self._agg: Dict[str, Dict[str, List[float]]] = {"train": {"loss": [], "acc": []}, "val": {"loss": [], "acc": []}}
        self._last_lr: Optional[float] = None
        self.meta: Dict[str, Any] = {
            "id": self.id,
            "name": name,
            "script": self._rel(self.script) if self.script else None,
            "cwd": os.getcwd(),
            "startedAt": _now_iso(),
            "endedAt": None,
            "status": "running",
            "pid": os.getpid(),
            "framework": None,
            "config": dict(config or {}),
            "model": {},
            "numClasses": None,
            "final": {},
            "best": {},
            "starred": False,
            "deepfriendVersion": __version__,
            "git": None,
            "record": record,
        }
        try:
            os.makedirs(self.dir, exist_ok=True)
            gi = os.path.join(base, ".gitignore")
            if not os.path.exists(gi):
                with open(gi, "w", encoding="utf-8") as f:
                    f.write("*\n")
            self.meta["git"] = _git_hash(self.root)
            self.write_meta()
            open(os.path.join(self.dir, "metrics.jsonl"), "a", encoding="utf-8").close()
            self.ok = True
        except Exception:
            self.ok = False
        if self.ok:
            self._thread = threading.Thread(target=self._flush_loop, name="deepfriend-writer", daemon=True)
            self._thread.start()
            atexit.register(self._atexit)

    # ---- paths ----

    def _rel(self, path: str) -> str:
        try:
            rel = os.path.relpath(path, self.root)
        except ValueError:  # different drive on Windows
            rel = path
        return rel.replace(os.sep, "/")

    # ---- writing ----

    def write_meta(self) -> None:
        try:
            # the VS Code extension may have starred the run meanwhile: keep that choice
            try:
                with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as f:
                    on_disk = json.load(f)
                if isinstance(on_disk, dict) and isinstance(on_disk.get("starred"), bool):
                    self.meta["starred"] = on_disk["starred"]
            except Exception:
                pass
            with self._lock:
                data = json.dumps(self.meta, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            _atomic_write(os.path.join(self.dir, "meta.json"), data)
        except Exception:
            pass

    def _emit(self, rec: Dict[str, Any]) -> None:
        if not self.ok:
            return
        try:
            line = json.dumps(rec, ensure_ascii=False, allow_nan=False)
        except ValueError:
            return
        with self._lock:
            self._pending.append(line)

    def flush(self) -> None:
        if not self.ok:
            return
        with self._lock:
            lines, self._pending = self._pending, []
        if not lines:
            return
        try:
            with open(os.path.join(self.dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(FLUSH_SECONDS):
            self.flush()

    # ---- recording API used by hooks ----

    def set_config(self, key: str, value: Any, overwrite: bool = False) -> None:
        with self._lock:
            if overwrite or key not in self.meta["config"]:
                self.meta["config"][key] = value

    def set_framework(self, name: str) -> None:
        if self.meta["framework"] is None:
            self.meta["framework"] = name
            self.write_meta()

    def train_step(self, loss: Any, acc: Optional[float] = None, lr: Optional[float] = None) -> None:
        try:
            with self._lock:
                if self._saw_val_since_train:
                    self._new_epoch()
                self.step += 1
                self._steps_in_epoch += 1
                rec: Dict[str, Any] = {"t": round(time.time(), 3), "step": self.step, "epoch": self.epoch, "split": "train"}
                f = _safe_float(loss)
                if f is None or not math.isfinite(f):
                    rec["loss"] = None
                    rec["nan"] = True
                else:
                    rec["loss"] = f
                    self._agg["train"]["loss"].append(f)
                if acc is not None:
                    rec["acc"] = acc
                    self._agg["train"]["acc"].append(acc)
                if lr is not None and lr != self._last_lr:
                    rec["lr"] = lr
                    self._last_lr = lr
                self._emit(rec)
        except Exception:
            pass

    def val_value(self, loss: Any, acc: Optional[float] = None) -> None:
        try:
            with self._lock:
                f = _safe_float(loss)
                if f is not None and math.isfinite(f):
                    self._agg["val"]["loss"].append(f)
                elif f is not None:
                    self._emit({"t": round(time.time(), 3), "step": self.step, "epoch": self.epoch, "split": "val", "loss": None, "nan": True})
                if acc is not None:
                    self._agg["val"]["acc"].append(acc)
                self._saw_val_since_train = True
        except Exception:
            pass

    def new_epoch_from_loader(self) -> None:
        with self._lock:
            if self._steps_in_epoch:
                self._new_epoch()

    def _new_epoch(self) -> None:
        self._flush_agg()
        self.epoch += 1
        self._steps_in_epoch = 0
        self._saw_val_since_train = False

    def _flush_agg(self) -> None:
        wrote = False
        for split in ("train", "val"):
            a = self._agg[split]
            if not a["loss"] and not a["acc"]:
                continue
            rec: Dict[str, Any] = {"t": round(time.time(), 3), "step": self.step, "epoch": self.epoch, "split": split, "agg": True}
            if a["loss"]:
                rec["loss"] = sum(a["loss"]) / len(a["loss"])
            if a["acc"]:
                rec["acc"] = sum(a["acc"]) / len(a["acc"])
            self._emit(rec)
            self._update_summary(split, rec)
            wrote = True
            a["loss"], a["acc"] = [], []
        if wrote:
            self.write_meta()

    def _update_summary(self, split: str, rec: Dict[str, Any]) -> None:
        prefix = "val_" if split == "val" else ""
        final = self.meta["final"]
        best = self.meta["best"]
        if "loss" in rec:
            final[prefix + "loss"] = rec["loss"]
            k = prefix + "loss"
            if k not in best or rec["loss"] < best[k]:
                best[k] = rec["loss"]
                best[k + "_epoch"] = rec["epoch"]
        if "acc" in rec:
            final[prefix + "acc"] = rec["acc"]
            k = prefix + "acc"
            if k not in best or rec["acc"] > best[k]:
                best[k] = rec["acc"]
                best[k + "_epoch"] = rec["epoch"]

    def log(self, step: Optional[int] = None, epoch: Optional[int] = None, **metrics: Any) -> None:
        """Manual logging: loss/acc are training values, val_loss/val_acc validation values."""
        try:
            with self._lock:
                if epoch is not None and epoch != self.epoch and (self._steps_in_epoch or self._agg["val"]["loss"]):
                    self._flush_agg()
                    self.epoch = epoch
                    self._steps_in_epoch = 0
                    self._saw_val_since_train = False
                elif epoch is not None:
                    self.epoch = epoch
            acc = metrics.pop("acc", metrics.pop("accuracy", None))
            loss = metrics.pop("loss", None)
            vloss = metrics.pop("val_loss", None)
            vacc = metrics.pop("val_acc", metrics.pop("val_accuracy", None))
            if loss is not None:
                if step is not None:
                    self.step = step - 1
                self.train_step(loss, _safe_float(acc) if acc is not None else None)
            if vloss is not None or vacc is not None:
                with self._lock:
                    rec: Dict[str, Any] = {"t": round(time.time(), 3), "step": self.step, "epoch": self.epoch, "split": "val", "agg": True}
                    if vloss is not None:
                        rec["loss"] = _safe_float(vloss)
                    if vacc is not None:
                        rec["acc"] = _safe_float(vacc)
                    self._emit(rec)
                    self._update_summary("val", rec)
                    self.write_meta()
            if metrics:
                with self._lock:
                    extra = {k: _safe_float(v) for k, v in metrics.items()}
                    self._emit({"t": round(time.time(), 3), "step": self.step, "epoch": self.epoch, "split": "user", **extra})
        except Exception:
            pass

    # ---- ending ----

    def stop(self, status: str = "finished") -> None:
        try:
            with self._lock:
                if self.status != "running":
                    return
                self.status = status
                self._flush_agg()
                self.meta["status"] = status
                self.meta["endedAt"] = _now_iso()
            self._stop_event.set()
            self.flush()
            self.write_meta()
            if not self.record:
                shutil.rmtree(self.dir, ignore_errors=True)
        except Exception:
            pass

    def _atexit(self) -> None:
        try:
            from . import snapshot

            snapshot.capture(self)
        except Exception:
            pass
        self.stop("finished")
