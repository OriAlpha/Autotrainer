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

    payload = json.dumps(
        {"old_run_id": "old_run_name", "new_run_id": "renamed_cpu_experiment"}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/runs/rename", data=payload, headers={"Content-Type": "application/json"}
    )
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
    req_add = urllib.request.Request(
        f"{base_url}/api/sources", data=add_payload, headers={"Content-Type": "application/json"}
    )
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
    req_del = urllib.request.Request(
        f"{base_url}/api/sources",
        data=del_payload,
        headers={"Content-Type": "application/json"},
        method="DELETE",
    )
    with urllib.request.urlopen(req_del) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["success"] is True


def test_ui_api_hardware_endpoint(ui_server):
    base_url, _ = ui_server
    req = urllib.request.Request(f"{base_url}/api/hardware")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "cpu_pct" in data
        assert "ram_used_gb" in data
        assert "ram_total_gb" in data
        assert "gpu_available" in data
        assert "gpus" in data


def test_ui_api_tags_endpoint(ui_server):
    base_url, logs_dir = ui_server
    tracker = NativeTracker(base_dir=logs_dir, run_name="run_tags_test")
    tracker.log_epoch(1, {"train_loss": 0.4})
    tracker.close()

    # 1. Add tag "baseline"
    payload = json.dumps({"action": "add", "tag": "baseline"}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/runs/run_tags_test/tags",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["success"] is True
        assert "baseline" in data["tags"]

    # 2. Add tag "lr-tuned"
    payload2 = json.dumps({"action": "add", "tag": "lr-tuned"}).encode("utf-8")
    req2 = urllib.request.Request(
        f"{base_url}/api/runs/run_tags_test/tags",
        data=payload2,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req2) as resp:
        assert resp.status == 200
        data2 = json.loads(resp.read().decode("utf-8"))
        assert "baseline" in data2["tags"]
        assert "lr-tuned" in data2["tags"]

    # 3. Remove tag "baseline"
    payload3 = json.dumps({"action": "remove", "tag": "baseline"}).encode("utf-8")
    req3 = urllib.request.Request(
        f"{base_url}/api/runs/run_tags_test/tags",
        data=payload3,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req3) as resp:
        assert resp.status == 200
        data3 = json.loads(resp.read().decode("utf-8"))
        assert "baseline" not in data3["tags"]
        assert "lr-tuned" in data3["tags"]


def test_ui_api_favorite_endpoint(ui_server):
    base_url, logs_dir = ui_server
    tracker = NativeTracker(base_dir=logs_dir, run_name="run_fav_test")
    tracker.log_epoch(1, {"train_loss": 0.3})
    tracker.close()

    # 1. Toggle favorite -> True
    payload = json.dumps({}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/runs/run_fav_test/favorite",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["success"] is True
        assert data["favorite"] is True

    # 2. Toggle favorite -> False
    req2 = urllib.request.Request(
        f"{base_url}/api/runs/run_fav_test/favorite",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req2) as resp:
        assert resp.status == 200
        data2 = json.loads(resp.read().decode("utf-8"))
        assert data2["favorite"] is False


def test_ui_api_notes_endpoint(ui_server):
    base_url, logs_dir = ui_server
    tracker = NativeTracker(base_dir=logs_dir, run_name="run_notes_test")
    tracker.log_epoch(1, {"train_loss": 0.2})
    tracker.close()

    notes_text = "# Experiment Takeaways\n- Achieved lowest loss with AdamW\n- Checkpoint verified"
    payload = json.dumps({"notes": notes_text}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/runs/run_notes_test/notes",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["success"] is True
        assert data["notes"] == notes_text

    # Verify notes.md file on disk
    notes_file = logs_dir / "run_notes_test" / "notes.md"
    assert notes_file.exists()
    assert notes_file.read_text(encoding="utf-8") == notes_text


def test_ui_api_delete_run_endpoint(ui_server):
    base_url, logs_dir = ui_server
    tracker = NativeTracker(base_dir=logs_dir, run_name="run_to_delete")
    tracker.log_epoch(1, {"train_loss": 0.5})
    tracker.close()

    assert (logs_dir / "run_to_delete").exists()

    req = urllib.request.Request(f"{base_url}/api/runs/run_to_delete", method="DELETE")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["success"] is True

    # Verify folder is deleted
    assert not (logs_dir / "run_to_delete").exists()


def test_native_tracker_metadata_features(tmp_path: Path):
    tracker = NativeTracker(base_dir=tmp_path, run_name="meta_features_run")
    tracker.add_tag("vit")
    tracker.add_tag("fp16")
    assert tracker.metadata["tags"] == ["vit", "fp16"]
    tracker.remove_tag("fp16")
    assert tracker.metadata["tags"] == ["vit"]

    tracker.set_favorite(True)
    assert tracker.metadata["favorite"] is True

    tracker.set_notes("Hypothesis: ViT converges faster with cosine schedule.")
    assert "cosine" in tracker.metadata["notes"]

    tracker.set_archived(True)
    assert tracker.metadata["archived"] is True
    tracker.close()

    # Re-open and verify persistence in run.json and notes.md
    meta_json = tmp_path / "meta_features_run" / "run.json"
    assert meta_json.exists()
    meta_data = json.loads(meta_json.read_text(encoding="utf-8"))
    assert meta_data["tags"] == ["vit"]
    assert meta_data["favorite"] is True
    assert meta_data["archived"] is True
    assert "cosine" in meta_data["notes"]

    notes_file = tmp_path / "meta_features_run" / "notes.md"
    assert notes_file.exists()
    assert "cosine" in notes_file.read_text(encoding="utf-8")
