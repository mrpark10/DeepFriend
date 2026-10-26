"""DeepFriend analysis server.

Started by the VS Code extension as ``python -u -m deepfriend.server``.
Speaks line-delimited JSON on stdin/stdout; stderr is for logs only.
"""

import importlib.util
import sys
import threading
import traceback
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

from . import __version__
from .protocol import read_messages, write_message

Handler = Callable[[Dict[str, Any]], Any]


def _package_version(name: str) -> Optional[str]:
    """Version of an installed package without importing it (importing torch takes seconds)."""
    try:
        if importlib.util.find_spec(name) is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return "unknown"


def log(*args: Any) -> None:
    print("[deepfriend]", *args, file=sys.stderr, flush=True)


class Server:
    def __init__(self, stdin=None, stdout=None) -> None:
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer
        self.handlers: Dict[str, Handler] = {}
        self.running = True
        self.register("initialize", self._initialize)
        self.register("shutdown", self._shutdown)
        self.register("echo", lambda params: params)
        self.register("findArgument", self._find_argument)
        # analyze requests run on a worker thread; a newer request for the same
        # uri replaces an older one that has not started yet ("latest wins").
        self._pending: "OrderedDict[str, Tuple[Any, Dict[str, Any]]]" = OrderedDict()
        self._cv = threading.Condition()
        self._analyzer: Any = None
        threading.Thread(target=self._analyze_loop, daemon=True).start()

    def register(self, method: str, fn: Handler) -> None:
        self.handlers[method] = fn

    def respond(self, msg_id: Any, result: Any = None, error: Optional[Dict[str, str]] = None) -> None:
        if error is not None:
            write_message(self.stdout, {"id": msg_id, "error": error})
        else:
            write_message(self.stdout, {"id": msg_id, "result": result})

    def handle(self, msg: Dict[str, Any]) -> None:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "analyze":
            self._enqueue_analyze(msg_id, params)
            return
        fn = self.handlers.get(method) if isinstance(method, str) else None
        if fn is None:
            self.respond(msg_id, error={"message": "unknown method: %s" % method, "trace": ""})
            return
        try:
            result = fn(params)
        except Exception as e:  # never let one request kill the server
            self.respond(msg_id, error={"message": "%s: %s" % (type(e).__name__, e), "trace": traceback.format_exc()})
            return
        self.respond(msg_id, result)

    def serve(self) -> None:
        for msg in read_messages(self.stdin):
            self.handle(msg)
            if not self.running:
                break

    def _enqueue_analyze(self, msg_id: Any, params: Dict[str, Any]) -> None:
        uri = str(params.get("uri", ""))
        with self._cv:
            old = self._pending.pop(uri, None)
            if old is not None:
                write_message(self.stdout, {"id": old[0], "result": None, "cancelled": True})
            self._pending[uri] = (msg_id, params)
            self._cv.notify()

    def _analyze_loop(self) -> None:
        while True:
            with self._cv:
                while not self._pending:
                    self._cv.wait()
                _uri, (msg_id, params) = self._pending.popitem(last=False)
            try:
                if self._analyzer is None:
                    from .shapes.analyze import Analyzer

                    self._analyzer = Analyzer()
                result = self._analyzer.analyze(str(params.get("uri", "")), str(params.get("text", "")), bool(params.get("runtime", False)))
            except Exception as e:
                self.respond(msg_id, error={"message": "%s: %s" % (type(e).__name__, e), "trace": traceback.format_exc()})
                continue
            self.respond(msg_id, result)

    def _find_argument(self, params: Dict[str, Any]) -> Any:
        from .findargs import find_argument

        return find_argument(str(params.get("text", "")), str(params.get("kind", "")))

    def _initialize(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "version": __version__,
            "python": sys.executable,
            "pythonVersion": "%d.%d.%d" % sys.version_info[:3],
            "torch": _package_version("torch"),
            "tensorflow": _package_version("tensorflow"),
            "torchImported": "torch" in sys.modules,
        }

    def _shutdown(self, _params: Dict[str, Any]) -> None:
        self.running = False
        if self._analyzer is not None:
            self._analyzer.close()
        return None


def main() -> None:
    server = Server()
    log("server started, python", sys.version.split()[0])
    try:
        server.serve()
    finally:
        if server._analyzer is not None:
            server._analyzer.close()


if __name__ == "__main__":
    main()
