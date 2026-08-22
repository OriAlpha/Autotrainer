"""Security regression tests for the Web UI's HTTP API.

Every case here corresponds to a vulnerability that was live in the first
release of ``autotrainer ui`` and was confirmed by exploiting a running
server, not by reading the code:

* ``run_id`` was pasted straight into ``root / run_id`` with no containment
  check, so ``GET /api/runs/../<dir>`` read outside the logs root and
  ``DELETE /api/runs/../<dir>`` reached ``shutil.rmtree`` on it. A live
  server returned ``{"success": true}`` and the directory was gone.
* The server bound ``0.0.0.0`` with no authentication, which on a shared
  SLURM login node published rename/archive/delete to the whole cluster.
* ``POST /api/sources`` accepted any absolute path, turning the dashboard
  into a filesystem browser (a home directory listed 45 "runs").
* The HTML export and Markdown export wrote ``report.html`` / ``report.md``
  through the same unchecked join - an arbitrary-write primitive.

These are cheap to keep and expensive to rediscover, so each one is pinned
directly rather than through the UI.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

from autotrainer.ui import AutotrainerUIHandler, _looks_like_logs_dir


def _make_run(root: Path, run_id: str) -> Path:
    run = root / run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    return run


@pytest.fixture
def secured_server(tmp_path: Path):
    """A server with a token, plus a sibling directory outside the logs root.

    The sibling is the thing traversal would reach; the tests assert it stays
    untouched and unreadable.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    _make_run(logs, "run_a")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run.json").write_text(json.dumps({"run_id": "victim"}), encoding="utf-8")
    (outside / "secret.txt").write_text("private", encoding="utf-8")

    AutotrainerUIHandler.logs_dirs = [logs]
    AutotrainerUIHandler.auth_token = "test-token-abc"

    server = HTTPServer(("127.0.0.1", 0), AutotrainerUIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}", logs, outside

    server.shutdown()
    server.server_close()
    AutotrainerUIHandler.auth_token = None


def _request(url: str, method: str = "GET", token: str | None = "test-token-abc", data=None):
    """Return (status, body). Errors come back as a status too, not a raise."""
    req = urllib.request.Request(url, method=method)
    if token:
        req.add_header("X-Autotrainer-Token", token)
    if data is not None:
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class TestPathTraversal:
    """``run_id`` must never escape a configured logs root."""

    @pytest.mark.parametrize(
        "run_id",
        [
            "../outside",
            "../../outside",
            "..%2Foutside",
            "run_a/../../outside",
            "..\\outside",
            "/etc",
        ],
    )
    def test_traversal_read_is_refused(self, secured_server, run_id):
        base, _logs, _outside = secured_server
        status, _ = _request(f"{base}/api/runs/{run_id}")
        assert status == 404, f"{run_id!r} was not rejected"

    def test_traversal_delete_does_not_remove_outside_directory(self, secured_server):
        """The worst of them: this used to rmtree whatever it resolved to."""
        base, _logs, outside = secured_server
        assert outside.exists()

        status, _ = _request(f"{base}/api/runs/../outside", method="DELETE")

        assert status == 404
        assert outside.exists(), "traversal deleted a directory outside the logs root"
        assert (outside / "secret.txt").exists()

    def test_legitimate_run_still_reachable(self, secured_server):
        """The guard must not break the normal case it wraps."""
        base, _logs, _outside = secured_server
        status, body = _request(f"{base}/api/runs/run_a")
        assert status == 200
        assert json.loads(body)["run_id"] == "run_a"

    def test_legitimate_delete_still_works(self, secured_server):
        base, logs, _outside = secured_server
        _make_run(logs, "run_doomed")
        status, _ = _request(f"{base}/api/runs/run_doomed", method="DELETE")
        assert status == 200
        assert not (logs / "run_doomed").exists()


class TestAuthentication:
    """With a token set, every endpoint requires it."""

    @pytest.mark.parametrize(
        ("path", "method"),
        [
            ("/api/runs", "GET"),
            ("/api/runs/run_a", "GET"),
            ("/api/sources", "GET"),
            ("/api/hardware", "GET"),
            ("/api/runs/run_a", "DELETE"),
        ],
    )
    def test_requests_without_token_are_rejected(self, secured_server, path, method):
        base, _logs, _outside = secured_server
        status, _ = _request(f"{base}{path}", method=method, token=None)
        assert status == 401

    def test_wrong_token_is_rejected(self, secured_server):
        base, _logs, _outside = secured_server
        status, _ = _request(f"{base}/api/runs", token="not-the-token")
        assert status == 401

    def test_token_in_query_string_is_accepted(self, secured_server):
        """How the printed URL arrives; routing must ignore the query."""
        base, _logs, _outside = secured_server
        status, _ = _request(f"{base}/api/runs?token=test-token-abc", token=None)
        assert status == 200

    def test_unauthenticated_delete_leaves_the_run_alone(self, secured_server):
        base, logs, _outside = secured_server
        _make_run(logs, "run_b")
        status, _ = _request(f"{base}/api/runs/run_b", method="DELETE", token=None)
        assert status == 401
        assert (logs / "run_b").exists()


class TestSourcePathRestriction:
    """``POST /api/sources`` must not accept arbitrary directories."""

    def test_arbitrary_directory_is_refused(self, secured_server, tmp_path):
        base, _logs, _outside = secured_server
        stranger = tmp_path / "not_logs"
        stranger.mkdir()
        (stranger / "personal.txt").write_text("x", encoding="utf-8")

        status, _ = _request(f"{base}/api/sources", method="POST", data={"path": str(stranger)})

        assert status == 400
        _, body = _request(f"{base}/api/sources")
        assert str(stranger) not in body

    def test_directory_holding_runs_is_accepted(self, secured_server, tmp_path):
        base, _logs, _outside = secured_server
        extra = tmp_path / "more_logs"
        _make_run(extra, "run_x")

        status, _ = _request(f"{base}/api/sources", method="POST", data={"path": str(extra)})
        assert status == 200

    def test_nonexistent_path_is_refused(self, secured_server, tmp_path):
        base, _logs, _outside = secured_server
        status, _ = _request(
            f"{base}/api/sources", method="POST", data={"path": str(tmp_path / "nope")}
        )
        assert status == 400


class TestLooksLikeLogsDir:
    def test_directory_with_a_run_marker(self, tmp_path):
        _make_run(tmp_path, "run_1")
        assert _looks_like_logs_dir(tmp_path)

    def test_directory_without_runs(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        assert not _looks_like_logs_dir(tmp_path)

    def test_missing_directory(self, tmp_path):
        assert not _looks_like_logs_dir(tmp_path / "absent")


class TestServerBindsLoopbackByDefault:
    def test_default_host_is_loopback(self):
        """The API deletes directories; the default must not be 0.0.0.0."""
        import inspect

        from autotrainer.ui import run_ui_server

        params = inspect.signature(run_ui_server).parameters
        assert params["host"].default == "127.0.0.1"

    def test_token_is_generated_by_default(self):
        import inspect

        from autotrainer.ui import run_ui_server

        params = inspect.signature(run_ui_server).parameters
        # "" means "generate one"; None would mean "no auth".
        assert params["token"].default == ""
