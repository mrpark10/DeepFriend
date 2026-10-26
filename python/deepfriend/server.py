"""DeepFriend analysis server.

Started by the VS Code extension as ``python -u -m deepfriend.server``.
Speaks line-delimited JSON on stdin/stdout; stderr is for logs only.
"""

import importlib.util
import sys
import traceback
from typing import Any, Callable, Dict, Optional

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
        return None


def main() -> None:
    server = Server()
    log("server started, python", sys.version.split()[0])
    server.serve()


if __name__ == "__main__":
    main()
