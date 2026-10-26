"""Runtime shape check on PyTorch's ``meta`` device (no real memory, no GPU).

``Worker`` (client side, used by the analysis server) talks to a long-lived
child process started with ``python -m deepfriend.shapes.runtime_torch``.
The child imports torch once, then for each request:

1. executes only the safe parts of the file (see ``extract.py``),
2. builds the model under ``torch.device("meta")``,
3. runs ``forward`` with ``sys.settrace`` recording the shape assigned on each
   line, plus forward hooks recording every submodule's output.

The child installs a best-effort sandbox (no file writes, no network, no
subprocesses, no ``torch.load``) because model constructors are user code.
Any hang is cut off by the client's timeout, which kills the child.
"""

import json
import os
import queue
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deepfriend.procutil import no_window_kwargs  # noqa: E402

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- client -----------------------------------------------------------------------


class Worker:
    def __init__(self, python: Optional[str] = None, ready_timeout: float = 60.0) -> None:
        self.python = python or sys.executable
        self.ready_timeout = ready_timeout
        self.proc: Optional[subprocess.Popen] = None
        self.lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self.next_id = 0
        self.torch_version: Optional[str] = None

    def _start(self) -> bool:
        env = dict(os.environ)
        env["PYTHONPATH"] = PKG_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self.proc = subprocess.Popen(
                [self.python, "-u", "-m", "deepfriend.shapes.runtime_torch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                **no_window_kwargs(),
            )
        except OSError:
            self.proc = None
            return False
        self.lines = queue.Queue()
        proc = self.proc
        q = self.lines

        def pump() -> None:
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                q.put(raw.decode("utf-8", errors="replace"))
            q.put(None)

        threading.Thread(target=pump, daemon=True).start()
        msg = self._read(self.ready_timeout)
        if not msg or not msg.get("ready"):
            self._kill()
            return False
        self.torch_version = msg.get("torch")
        return True

    def _read(self, timeout: float) -> Optional[Dict[str, Any]]:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            return None
        try:
            return json.loads(line)
        except ValueError:
            return None

    def _kill(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self.proc = None

    def check(self, text: str, model: str, inputs: Sequence[Tuple[Any, ...]], timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """Return the runtime result, or None when the check cannot be trusted."""
        if self.proc is None or self.proc.poll() is not None:
            if not self._start():
                return None
        assert self.proc is not None and self.proc.stdin is not None
        self.next_id += 1
        req = {"id": self.next_id, "text": text, "model": model, "inputs": [list(s) for s in inputs]}
        try:
            self.proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except OSError:
            self._kill()
            return None
        while True:
            msg = self._read(timeout)
            if msg is None:
                self._kill()
                return None
            if msg.get("id") == self.next_id:
                break
        if msg.get("status") != "ok":
            return None
        return msg["result"]

    def close(self) -> None:
        self._kill()


# ---- child process ------------------------------------------------------------------

SHAPE_ERROR_PATTERNS = [
    "mat1 and mat2 shapes",
    "size mismatch",
    "shapes cannot be multiplied",
    "Given groups=",
    "expected input",
    "must match the size of tensor",
    "The size of tensor a",
    "is invalid for input of size",
    "Sizes of tensors must match",
    "Kernel size can't be greater",
    "Calculated output size",
    "Expected 2D (unbatched) or 3D",
    "Expected 3D (unbatched) or 4D",
    "Expected 4D (unbatched) or 5D",
    "expected 2D or 3D input",
    "expected 4D input",
    "running_mean should contain",
    "Expected input batch_size",
    "doesn't match the broadcast shape",
    "Dimension out of range",
    "number of dims don't match in permute",
    "size of dimension",
    "Trying to create tensor with negative dimension",
    "Output size is too small",
    "pad should be",
    "padding should be",
    # meta-device wording (differs from CPU)
    "must have same reduction dim",
    "Invalid channel dimensions",
    "Attempting to broadcast a dimension",
    "Expected size for first two dimensions",
    "expected input with shape",
    "does not match the length of the desired ordering",
    "expects each tensor to be equal size",
    "must be equal to input_size",
    "was expected to have embed_dim",
]


def _blocked(*_a: Any, **_k: Any) -> Any:
    raise PermissionError("DeepFriend: this action is blocked during the shape check")


def _install_sandbox(torch: Any) -> None:
    import builtins
    import io
    import shutil
    import socket

    real_open = builtins.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, int) or any(c in mode for c in "wax+"):
            raise PermissionError("DeepFriend: writing files is blocked during the shape check")
        return real_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open  # type: ignore[assignment]
    io.open = guarded_open  # type: ignore[assignment]
    socket.socket.connect = _blocked  # type: ignore[assignment]
    socket.socket.connect_ex = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    for name in ("system", "remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace", "kill", "fork", "execv", "execve", "spawnv", "popen"):
        if hasattr(os, name):
            setattr(os, name, _blocked)
    shutil.rmtree = _blocked  # type: ignore[assignment]
    shutil.move = _blocked  # type: ignore[assignment]
    subprocess.Popen = _blocked  # type: ignore[assignment,misc]
    subprocess.run = _blocked  # type: ignore[assignment]
    subprocess.call = _blocked  # type: ignore[assignment]
    torch.load = _blocked
    torch.save = _blocked
    try:
        torch.hub.load = _blocked
        torch.hub.load_state_dict_from_url = _blocked
        torch.hub.download_url_to_file = _blocked
    except AttributeError:
        pass


def _warm_up(torch: Any) -> None:
    """Trigger torch's lazy imports (dynamo, decompositions, temp dir probe) before the sandbox."""
    import tempfile

    tempfile.gettempdir()
    try:
        import torch._dynamo  # noqa: F401
    except Exception:
        pass
    nn = torch.nn
    try:
        with torch.device("meta"):
            layers = [nn.Linear(4, 4), nn.Conv2d(3, 4, 3), nn.Conv1d(3, 4, 3), nn.BatchNorm2d(4), nn.LayerNorm(4), nn.Embedding(10, 4), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1), nn.LSTM(4, 4, batch_first=True)]
        x2 = torch.empty(2, 4, device="meta")
        layers[0](x2) @ x2.T
        layers[1](torch.empty(1, 3, 8, 8, device="meta"))
        layers[2](torch.empty(1, 3, 8, device="meta"))
        layers[3].eval()(torch.empty(1, 4, 6, 6, device="meta"))
        layers[4](x2)
        layers[5](torch.empty(2, 3, dtype=torch.long, device="meta"))
        layers[6](torch.empty(1, 4, 6, 6, device="meta"))
        layers[7](torch.empty(1, 4, 6, 6, device="meta"))
        layers[8](torch.empty(1, 3, 4, device="meta"))
        torch.cat([x2, x2], 1).softmax(-1).view(-1).reshape(2, 8)
    except Exception:
        pass


def _statement_table(tree: Any) -> Dict[int, Tuple[Optional[List[str]], int, bool]]:
    """start line -> (assigned names or None, last line, is_return) for every statement in functions."""
    import ast

    table: Dict[int, Tuple[Optional[List[str]], int, bool]] = {}

    def names_of(t: Any) -> Optional[List[str]]:
        if isinstance(t, ast.Name):
            return [t.id]
        if isinstance(t, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Name) for e in t.elts):
            return [e.id for e in t.elts]
        return None

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for st in ast.walk(fn):
            if not isinstance(st, ast.stmt) or st is fn:
                continue
            start, end = st.lineno - 1, (st.end_lineno or st.lineno) - 1
            if isinstance(st, ast.Assign):
                table[start] = (names_of(st.targets[0]) if len(st.targets) == 1 else None, end, False)
            elif isinstance(st, (ast.AugAssign, ast.AnnAssign)):
                table[start] = (names_of(st.target), end, False)
            elif isinstance(st, ast.Return):
                table[start] = (None, end, True)
            elif hasattr(st, "body"):
                table[start] = (None, start, False)  # compound statement: header only
            else:
                table[start] = (None, end, False)
    return table


def _fmt(torch: Any, v: Any) -> Optional[str]:
    from deepfriend.shapes.dims import fmt_shape

    if torch.is_tensor(v):
        return fmt_shape(tuple(int(d) for d in v.shape))
    return None


def _check(torch: Any, req: Dict[str, Any]) -> Dict[str, Any]:
    import ast
    import inspect
    import traceback

    from deepfriend.shapes.extract import safe_source

    text, model_name = req["text"], req["model"]
    inputs = [tuple(2 if not isinstance(d, int) else d for d in s) for s in req["inputs"]]
    tree = ast.parse(text)
    table = _statement_table(tree)
    code = compile(safe_source(tree), "<deepfriend>", "exec")
    ns: Dict[str, Any] = {"__name__": "deepfriend_check"}
    exec(code, ns)
    cls = ns.get(model_name)
    if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
        return {"status": "skip", "reason": "not_a_module"}
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {"status": "skip", "reason": "signature"}
    for p in list(sig.parameters.values())[1:]:
        if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            return {"status": "skip", "reason": "ctor_args"}
    with torch.device("meta"):
        model = cls()
    model.eval()

    module_out: Dict[str, List[str]] = {}
    handles = []
    for name, mod in model.named_modules():
        if not name:
            continue

        def hook(_m: Any, _i: Any, out: Any, name: str = name) -> None:
            s = _fmt(torch, out)
            if s is not None:
                lst = module_out.setdefault(name, [])
                if s not in lst:
                    lst.append(s)

        handles.append(mod.register_forward_hook(hook))

    lines: Dict[int, List[str]] = {}

    def add(line: int, s: str) -> None:
        lst = lines.setdefault(line, [])
        if s not in lst:
            lst.append(s)

    def flush(prev: int, frame: Any) -> None:
        names, end, _ret = table[prev]
        if not names:
            return
        shapes = [_fmt(torch, frame.f_locals.get(n)) for n in names]
        if all(s is not None for s in shapes):
            add(end, ", ".join(shapes))  # type: ignore[arg-type]

    def global_tracer(frame: Any, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename != "<deepfriend>":
            return None
        state: Dict[str, Any] = {"prev": None}

        def local(frame: Any, event: str, arg: Any) -> Any:
            if event == "line":
                ln = frame.f_lineno - 1
                prev = state["prev"]
                if prev is not None:
                    _names, end, _r = table[prev]
                    if prev <= ln <= end and ln != prev:
                        return local  # still inside the same multi-line statement
                    flush(prev, frame)
                state["prev"] = ln if ln in table else None
            elif event == "return":
                prev = state["prev"]
                if prev is not None:
                    names, end, is_ret = table[prev]
                    if is_ret:
                        s = _fmt(torch, arg)
                        if s is not None:
                            add(end, s)
                    else:
                        flush(prev, frame)
                state["prev"] = None
            return local

        return local

    def run(dtype: Any) -> Any:
        xs = [torch.empty(s, dtype=dtype, device="meta") for s in inputs]
        sys.settrace(global_tracer)
        try:
            with torch.no_grad():
                return model(*xs)
        finally:
            sys.settrace(None)

    error = None
    try:
        try:
            run(torch.float32)
        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ("Long", "Int", "indices", "embedding")):
                lines.clear()
                module_out.clear()
                run(torch.long)
            else:
                raise
    except Exception as e:
        msg = str(e)
        if not any(p in msg for p in SHAPE_ERROR_PATTERNS):
            return {"status": "skip", "reason": "runtime:" + type(e).__name__}
        line = None
        for fr, lineno in traceback.walk_tb(e.__traceback__):
            if fr.f_code.co_filename == "<deepfriend>":
                line = lineno - 1
        error = {"line": line, "message": msg.splitlines()[0][:300], "type": type(e).__name__}
    finally:
        for h in handles:
            h.remove()

    modules = {}
    for name, mod in model.named_modules():
        if name:
            modules[name] = {"out": module_out.get(name, []), "params": sum(p.numel() for p in mod.parameters())}
    result = {
        "lines": [[ln, shapes] for ln, shapes in sorted(lines.items())],
        "modules": modules,
        "totalParams": sum(p.numel() for p in model.parameters()),
        "error": error,
    }
    return {"status": "ok", "result": result}


def child_main() -> None:
    out = sys.stdout.buffer
    try:
        import torch
    except Exception:
        out.write(b'{"ready": false}\n')
        out.flush()
        return
    _warm_up(torch)
    _install_sandbox(torch)
    out.write((json.dumps({"ready": True, "torch": torch.__version__}) + "\n").encode("utf-8"))
    out.flush()
    for raw in iter(sys.stdin.buffer.readline, b""):
        try:
            req = json.loads(raw.decode("utf-8"))
        except ValueError:
            continue
        try:
            resp = _check(torch, req)
        except BaseException as e:  # SystemExit from user code must not kill the worker
            if isinstance(e, KeyboardInterrupt):
                raise
            resp = {"status": "skip", "reason": "exception:" + type(e).__name__}
        resp["id"] = req.get("id")
        out.write((json.dumps(resp) + "\n").encode("utf-8"))
        out.flush()


if __name__ == "__main__":
    child_main()
