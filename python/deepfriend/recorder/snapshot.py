"""Copy the running script and the project's own imported .py files into run/code/."""

import hashlib
import os
import shutil
import sys
from typing import Any, Dict

MAX_TOTAL = 5 * 1024 * 1024
_SKIP_PARTS = ("site-packages", "dist-packages", ".venv", "venv", "env", ".deepfriend", "__pycache__", "node_modules")


def _project_files(run: Any) -> Dict[str, str]:
    files: Dict[str, str] = {}
    root = os.path.abspath(run.root)
    candidates = [run.script] if run.script else []
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".py"):
            candidates.append(os.path.abspath(f))
    for f in candidates:
        if not f or not os.path.isfile(f):
            continue
        try:
            rel = os.path.relpath(f, root)
        except ValueError:
            continue
        if rel.startswith("..") or any(p in rel.replace("\\", "/").split("/") for p in _SKIP_PARTS):
            continue
        files[rel.replace(os.sep, "/")] = f
    return files


def capture(run: Any) -> None:
    """Snapshot project files (≤5 MB). Called at watch() and again at exit; unchanged files are skipped."""
    if not getattr(run, "ok", False):
        return
    code_dir = os.path.join(run.dir, "code")
    total = 0
    for rel, src in sorted(_project_files(run).items()):
        try:
            size = os.path.getsize(src)
            if total + size > MAX_TOTAL:
                break
            dst = os.path.join(code_dir, *rel.split("/"))
            if os.path.exists(dst):
                with open(src, "rb") as a, open(dst, "rb") as b:
                    if hashlib.sha1(a.read()).digest() == hashlib.sha1(b.read()).digest():
                        total += size
                        continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            total += size
        except OSError:
            continue
