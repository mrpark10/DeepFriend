"""Subprocess helpers shared by the analyzer and the recorder."""

import subprocess
import sys
from typing import Any, Dict


def no_window_kwargs() -> Dict[str, Any]:
    """On Windows, keep child processes from opening a console window (the editor has none)."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}
