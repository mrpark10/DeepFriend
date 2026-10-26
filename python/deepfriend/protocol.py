"""Line-delimited JSON messages over binary streams (stdin/stdout).

Streams are used in binary mode and encoded as UTF-8 explicitly so the
protocol behaves the same on Windows (where the console code page may not be
UTF-8) and on macOS/Linux.
"""

import json
import threading
from typing import Any, BinaryIO, Dict, Iterator

_write_lock = threading.Lock()


def read_messages(stream: BinaryIO) -> Iterator[Dict[str, Any]]:
    """Yield one dict per valid JSON line; malformed lines are skipped."""
    for raw in iter(stream.readline, b""):
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if isinstance(msg, dict):
            yield msg


def write_message(stream: BinaryIO, obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _write_lock:
        stream.write(data.encode("utf-8"))
        stream.flush()
