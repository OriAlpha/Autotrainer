"""Tests for autotrainer Web UI handler and API endpoints."""

import json
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path
import pytest
from autotrainer.trackers import NativeTracker
from autotrainer.ui import AutotrainerUIHandler


@pytest.fixture
def ui_server(tmp_path: Path):
    """Fixture that runs an AutotrainerUIHandler server on a random free port."""
    AutotrainerUIHandler.logs_dirs = [tmp_path]

    class ThreadedHTTPServer(HTTPServer):
        pass

    server = ThreadedHTTPServer(("127.0.0.1", 0), AutotrainerUIHandler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    yield f"http://127.0.0.1:{port}", tmp_path

    server.shutdown()
    server.server_close()


def test_ui_index_endpoint(ui_server):
    base_url, _ = ui_server
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "Autotrainer Web UI" in content


def test_ui_api_runs_endpoint(ui_server):
    base_url, logs_dir = ui_server
    tracker = NativeTracker(base_dir=logs_dir, run_name="run_ui_test")
    tracker.log_epoch(1, {"train_loss": 0.5})
    tracker.close()

    req = urllib.request.Request(f"{base_url}/api/runs")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert isinstance(data, list)
        assert any(r.get("run_id") == "run_ui_test" for r in data)


def test_ui_api_run_detail_endpoint(ui_server):
    base_url, logs_dir = ui_server
    tracker = NativeTracker(base_dir=logs_dir, run_name="run_detail_test")
    tracker.log_params({"lr": 0.01})
    tracker.log_epoch(1, {"train_loss": 0.4, "val_loss": 0.3})
    tracker.close()

    req = urllib.request.Request(f"{base_url}/api/runs/run_detail_test")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["run_id"] == "run_detail_test"
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["train_loss"] == 0.4
        assert "paths" in data


def test_ui_api_rename_endpoint(ui_server):
    base_url, logs_dir = ui_server
    tracker = NativeTracker(base_dir=logs_dir, run_name="old_run_name")
    tracker.log_epoch(1, {"train_loss": 0.5})
    tracker.close()

    payload = json.dumps({"old_run_id": "old_run_name", "new_run_id": "renamed_cpu_experiment"}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/runs/rename", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["success"] is True
        assert res["run_id"] == "renamed_cpu_experiment"

    # Verify folder on disk was renamed
    assert (logs_dir / "renamed_cpu_experiment").exists()
    assert not (logs_dir / "old_run_name").exists()


def test_ui_api_export_endpoint(ui_server):
    base_url, logs_dir = ui_server
    tracker = NativeTracker(base_dir=logs_dir, run_name="export_test_run", user="test_user")
    tracker.log_params({"lr": 0.001, "batch_size": 32})
    tracker.log_epoch(1, {"train_loss": 0.8, "val_loss": 0.7})
    tracker.log_epoch(2, {"train_loss": 0.5, "val_loss": 0.4})
    tracker.close()

    # 1. Test HTML Export
    req_html = urllib.request.Request(f"{base_url}/api/runs/export_test_run/export")
    with urllib.request.urlopen(req_html) as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers.get("Content-Type", "")
        content = resp.read().decode("utf-8")
        assert "Autotrainer Executive Report" in content
        assert "export_test_run" in content

    # 2. Test Markdown Export
    req_md = urllib.request.Request(f"{base_url}/api/runs/export_test_run/export/markdown")
    with urllib.request.urlopen(req_md) as resp:
        assert resp.status == 200
        assert "text/markdown" in resp.headers.get("Content-Type", "")
        content = resp.read().decode("utf-8")
        assert "# ⚡ Autotrainer Run Summary" in content
        assert "test_user" in content

    # 3. Test CSV Export
    req_csv = urllib.request.Request(f"{base_url}/api/runs/export_test_run/export/csv")
    with urllib.request.urlopen(req_csv) as resp:
        assert resp.status == 200
        assert "text/csv" in resp.headers.get("Content-Type", "")
        content = resp.read().decode("utf-8")
        assert "train_loss" in content

    # 4. Test JSON Export
    req_json = urllib.request.Request(f"{base_url}/api/runs/export_test_run/export/json")
    with urllib.request.urlopen(req_json) as resp:
        assert resp.status == 200
        assert "application/json" in resp.headers.get("Content-Type", "")
        data = json.loads(resp.read().decode("utf-8"))
        assert data["run_id"] == "export_test_run"


def test_ui_api_sources_and_multi_paths(ui_server, tmp_path_factory):
    base_url, default_dir = ui_server
    extra_dir = tmp_path_factory.mktemp("extra_logs")

    # Create run in extra directory
    tracker = NativeTracker(base_dir=extra_dir, run_name="run_in_extra_dir", user="worker_2")
    tracker.log_epoch(1, {"train_loss": 0.3})
    tracker.close()

    # Get sources initially
    req = urllib.request.Request(f"{base_url}/api/sources")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        sources = json.loads(resp.read().decode("utf-8"))
        assert len(sources) == 1
        assert sources[0]["is_default"] is True

    # Add extra path via POST /api/sources
    add_payload = json.dumps({"path": str(extra_dir)}).encode("utf-8")
    req_add = urllib.request.Request(f"{base_url}/api/sources", data=add_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_add) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["success"] is True

    # Verify both sources exist and runs from extra_dir are aggregated
    req_runs = urllib.request.Request(f"{base_url}/api/runs")
    with urllib.request.urlopen(req_runs) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert any(r["run_id"] == "run_in_extra_dir" for r in data)

    # Remove extra source via DELETE /api/sources
    del_payload = json.dumps({"path": str(extra_dir)}).encode("utf-8")
    req_del = urllib.request.Request(f"{base_url}/api/sources", data=del_payload, headers={"Content-Type": "application/json"}, method="DELETE")
    with urllib.request.urlopen(req_del) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["success"] is True
