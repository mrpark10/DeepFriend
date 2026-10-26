"""Entry point for F1/F4: static analysis, optional torch check, merge, UTF-16 columns."""

import ast
import copy
import hashlib
import sys
from typing import Any, Dict, List, Optional

from .dims import is_int
from .static_keras import analyze_keras
from .static_torch import _memory, analyze_torch
from .torch_layers import collect_aliases


def to_utf16_col(line_text: str, byte_col: int) -> int:
    """Convert an AST UTF-8 byte column to a VS Code (UTF-16) column."""
    prefix = line_text.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore")
    return sum(2 if ord(c) > 0xFFFF else 1 for c in prefix)


def detect_framework(tree: ast.Module) -> Optional[str]:
    tops = {v.split(".")[0] for v in collect_aliases(tree).values()}
    if "torch" in tops:
        return "pytorch"
    if tops & {"tensorflow", "keras", "tf_keras"}:
        return "tensorflow"
    return None


def _has_symbolic(shapes: List[str]) -> bool:
    return any(any(c.isalpha() for c in s) for s in shapes)


def _parse_shape(s: str) -> tuple:
    inner = s.strip()[1:-1]
    return tuple(int(p) if p.strip().lstrip("-").isdigit() else p.strip() for p in inner.split(",") if p.strip())


def merge(models: List[Dict[str, Any]], runtime: Optional[Dict[str, Any]], symbolic: bool, lines: List[str], owner: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fold a runtime result for ``owner`` into the static results (runtime wins)."""
    if runtime is None:
        return models
    rt_lines = {int(l): s for l, s in runtime["lines"]}
    err = runtime.get("error")
    targets = [m for m in models if m is owner or (m.get("input") is None and not m.get("needsInput") and m.get("kind") == "torch")]
    if not symbolic:
        # module outputs -> blocks and their definition lines
        for b in owner["blocks"]:
            info = runtime["modules"].get(b["name"])
            if info is None:
                continue
            if info["out"]:
                b["outShape"] = info["out"][0]
                if "_hintLine" in b:
                    rt_lines.setdefault(b["_hintLine"], info["out"])
            b["params"] = info["params"]
        for m in targets:
            hints = {h["line"]: h for h in m["hints"]}
            for line, shapes in rt_lines.items():
                if not m["line"] <= line <= m["endLine"]:
                    continue
                h: Dict[str, Any] = {"line": line, "shape": shapes[0]}
                if len(shapes) > 1:
                    h["all"] = list(shapes)
                hints[line] = h
            m["hints"] = [hints[k] for k in sorted(hints)]
            if err is None:
                m["diagnostics"] = []
        owner["totalParams"] = runtime["totalParams"]
        outs = [_parse_shape(b["outShape"]) if b["outShape"] != "?" else None for b in owner["blocks"]]
        owner["memory"] = _memory(owner["totalParams"], outs)
        owner["verified"] = True
    if err is not None and err.get("line") is not None:
        line = err["line"]
        holder = next((m for m in targets if m["line"] <= line <= m["endLine"]), owner)
        if not holder["diagnostics"] and 0 <= line < len(lines):
            text = lines[line]
            raw = text.encode("utf-8")
            indent = len(raw) - len(raw.lstrip())
            holder["diagnostics"].append({"line": line, "col": indent, "endLine": line, "endCol": len(raw), "code": "runtime_error", "args": {"message": err["message"]}})
    return models


def _convert_cols(models: List[Dict[str, Any]], lines: List[str]) -> None:
    def conv(line: int, col: int) -> int:
        return to_utf16_col(lines[line], col) if 0 <= line < len(lines) else col

    for m in models:
        for d in m["diagnostics"]:
            d["col"] = conv(d["line"], d["col"])
            d["endCol"] = conv(d["endLine"], d["endCol"])
            fix = d.get("fix")
            if fix and not fix.get("_converted"):
                fix["col"] = conv(fix["line"], fix["col"])
                fix["endCol"] = conv(fix["endLine"], fix["endCol"])
                fix["_converted"] = True
        for d in m["diagnostics"]:
            if d.get("fix"):
                d["fix"].pop("_converted", None)
        for b in m["blocks"]:
            b.pop("_hintLine", None)


class Analyzer:
    def __init__(self) -> None:
        self.last_good: Dict[str, Dict[str, Any]] = {}
        self.cache: Dict[str, Any] = {}
        self.worker: Any = None
        self.worker_failed = False

    def _worker(self) -> Any:
        if self.worker is None and not self.worker_failed:
            try:
                from .runtime_torch import Worker

                self.worker = Worker()
            except Exception:
                self.worker_failed = True
        return self.worker

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def analyze(self, uri: str, text: str, runtime: bool) -> Dict[str, Any]:
        key = hashlib.sha1((("1" if runtime else "0") + text).encode("utf-8", errors="replace")).hexdigest()
        cached = self.cache.get(uri)
        if cached and cached[0] == key:
            return copy.deepcopy(cached[1])
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as e:
            prev = self.last_good.get(uri)
            line = max(0, (getattr(e, "lineno", None) or 1) - 1)
            col = max(0, (getattr(e, "offset", None) or 1) - 1)
            res = copy.deepcopy(prev) if prev else {"framework": None, "models": []}
            res["syntaxError"] = {"line": line, "col": col, "msg": str(getattr(e, "msg", e))}
            return res
        lines = text.splitlines()
        framework = detect_framework(tree)
        models: List[Dict[str, Any]] = []
        if framework is not None:
            for fn in (analyze_torch, analyze_keras):
                try:
                    models.extend(fn(tree, text))
                except Exception as e:  # an analyzer bug must never break the editor
                    print("[deepfriend] analyzer error:", type(e).__name__, e, file=sys.stderr, flush=True)
        if runtime and framework == "pytorch":
            for m in [m for m in models if m["kind"] == "torch" and m["input"] is not None]:
                worker = self._worker()
                if worker is None:
                    break
                shapes = m["input"]["shapes"]
                try:
                    rt = worker.check(text, m["name"], [_parse_shape(s) for s in shapes])
                except Exception:
                    rt = None
                merge(models, rt, _has_symbolic(shapes), lines, m)
        _convert_cols(models, lines)
        res = {"framework": framework, "models": models, "syntaxError": None}
        self.last_good[uri] = res
        self.cache[uri] = (key, res)
        return copy.deepcopy(res)


__all__ = ["Analyzer", "merge", "to_utf16_col", "detect_framework", "is_int"]
