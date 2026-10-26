"""Record uncaught exceptions to error.json and mark the run failed/stopped."""

import json
import os
import sys
import threading
import traceback
from typing import Any


def _write(run: Any, etype: Any, value: Any, tb: Any) -> None:
    try:
        if issubclass(etype, KeyboardInterrupt):
            run.stop("stopped")
            return
        frames = [{"file": f.filename, "line": f.lineno, "name": f.name} for f in traceback.extract_tb(tb)]
        data = {
            "type": etype.__name__,
            "message": str(value),
            "traceback": "".join(traceback.format_exception(etype, value, tb)),
            "frames": frames,
        }
        if run.ok:
            with open(os.path.join(run.dir, "error.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        run.stop("failed")
    except Exception:
        pass


def install(run: Any) -> None:
    prev = sys.excepthook

    def hook(etype: Any, value: Any, tb: Any) -> None:
        _write(run, etype, value, tb)
        prev(etype, value, tb)

    sys.excepthook = hook

    if hasattr(threading, "excepthook"):
        prev_t = threading.excepthook

        def thook(args: Any) -> None:
            _write(run, args.exc_type, args.exc_value, args.exc_traceback)
            prev_t(args)

        threading.excepthook = thook

    try:  # Jupyter / IPython: uncaught errors do not reach sys.excepthook
        ip = sys.modules.get("IPython")
        shell = ip.get_ipython() if ip is not None and hasattr(ip, "get_ipython") else None
        if shell is not None:

            def ipy_handler(shell: Any, etype: Any, value: Any, tb: Any, tb_offset: Any = None) -> Any:
                _write_error_only(run, etype, value, tb)
                return shell.showtraceback((etype, value, tb), tb_offset=tb_offset)

            shell.set_custom_exc((Exception,), ipy_handler)
    except Exception:
        pass


def _write_error_only(run: Any, etype: Any, value: Any, tb: Any) -> None:
    """In notebooks an error does not end the run, so only error.json is written."""
    try:
        frames = [{"file": f.filename, "line": f.lineno, "name": f.name} for f in traceback.extract_tb(tb)]
        if run.ok:
            with open(os.path.join(run.dir, "error.json"), "w", encoding="utf-8") as f:
                json.dump({"type": etype.__name__, "message": str(value), "traceback": "", "frames": frames}, f, ensure_ascii=False)
    except Exception:
        pass
