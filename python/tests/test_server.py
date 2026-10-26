import json
import os
import subprocess
import sys

import pytest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ServerProc:
    def __init__(self):
        env = dict(os.environ, PYTHONPATH=PKG_ROOT, PYTHONIOENCODING="utf-8")
        self.p = subprocess.Popen(
            [sys.executable, "-u", "-m", "deepfriend.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    def send_raw(self, line: str):
        self.p.stdin.write(line.encode("utf-8"))
        self.p.stdin.flush()

    def send(self, obj):
        self.send_raw(json.dumps(obj) + "\n")

    def recv(self, timeout=10.0):
        line = self.p.stdout.readline()
        assert line, "server closed stdout; stderr=" + self.p.stderr.read().decode()
        return json.loads(line.decode("utf-8"))

    def request(self, id_, method, params=None):
        self.send({"id": id_, "method": method, "params": params or {}})
        return self.recv()

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()


@pytest.fixture
def server():
    s = ServerProc()
    yield s
    s.close()


def test_initialize_reports_versions(server):
    r = server.request(1, "initialize")
    assert r["id"] == 1
    res = r["result"]
    assert res["version"] == "0.1.0"
    assert res["pythonVersion"].startswith("3.")
    assert "torch" in res and "tensorflow" in res
    assert os.path.realpath(res["python"]) == os.path.realpath(sys.executable)


def test_initialize_does_not_import_torch(server):
    r = server.request(1, "initialize")
    assert r["result"]["torchImported"] is False


def test_unknown_method_returns_error(server):
    r = server.request(2, "nope")
    assert "nope" in r["error"]["message"]
    r2 = server.request(3, "initialize")
    assert r2["result"]["version"] == "0.1.0"


def test_bad_json_line_ignored(server):
    server.send_raw("{not json\n")
    r = server.request(4, "initialize")
    assert r["id"] == 4


def test_non_ascii_roundtrip(server):
    server.send({"id": 5, "method": "echo", "params": {"text": "박준호 딥러닝 🚀"}})
    r = server.recv()
    assert r["result"] == {"text": "박준호 딥러닝 🚀"}


def test_shutdown_exits(server):
    r = server.request(6, "shutdown")
    assert r["result"] is None
    server.p.wait(timeout=2)
    assert server.p.returncode == 0


def test_stdin_eof_exits(server):
    server.p.stdin.close()
    server.p.wait(timeout=2)
    assert server.p.returncode == 0
