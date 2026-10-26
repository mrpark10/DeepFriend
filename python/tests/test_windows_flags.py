import subprocess
import sys

from deepfriend.procutil import no_window_kwargs


def test_no_console_window_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    assert no_window_kwargs() == {"creationflags": 0x08000000}


def test_nothing_extra_elsewhere(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert no_window_kwargs() == {}


def test_worker_and_git_use_it():
    import inspect

    from deepfriend.recorder import run
    from deepfriend.shapes import runtime_torch

    assert "no_window_kwargs()" in inspect.getsource(runtime_torch.Worker._start)
    assert "no_window_kwargs()" in inspect.getsource(run._git_hash)
