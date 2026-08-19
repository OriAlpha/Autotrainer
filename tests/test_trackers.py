"""Tests for native experiment tracking and framework callbacks."""

import json
from pathlib import Path
from autotrainer.trackers import CSVTracker, JSONLTracker, NativeTracker
from autotrainer.callbacks import (
    AutotrainerKerasCallback,
    autotrainer_xgboost_callback,
    autotrainer_lightgbm_callback,
)


def test_csv_tracker(tmp_path: Path):
    tracker = CSVTracker(tmp_path, "metrics.csv")
    tracker.log_epoch(1, {"train_loss": 0.5, "val_loss": 0.4})
    tracker.log_epoch(2, {"train_loss": 0.3, "val_loss": 0.2})
    tracker.close()

    csv_path = tmp_path / "metrics.csv"
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "epoch" in content and "train_loss" in content and "val_loss" in content
    assert ",1,,0.5,0.4" in content
    assert ",2,,0.3,0.2" in content


def test_jsonl_tracker(tmp_path: Path):
    tracker = JSONLTracker(tmp_path, "metrics.jsonl")
    tracker.log_params({"lr": 0.001, "batch_size": 32})
    tracker.log_epoch(1, {"train_loss": 0.5})
    tracker.log_summary({"duration": 12.34})
    tracker.close()

    jsonl_path = tmp_path / "metrics.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    p_rec = json.loads(lines[0])
    assert p_rec["type"] == "params"
    assert p_rec["params"]["lr"] == 0.001

    e_rec = json.loads(lines[1])
    assert e_rec["type"] == "epoch"
    assert e_rec["epoch"] == 1
    assert e_rec["metrics"]["train_loss"] == 0.5

    s_rec = json.loads(lines[2])
    assert s_rec["type"] == "summary"
    assert s_rec["summary"]["duration"] == 12.34


def test_native_tracker(tmp_path: Path):
    tracker = NativeTracker(base_dir=tmp_path, run_name="test_run_01")
    tracker.log_params({"model": "resnet18"})
    tracker.log_epoch(1, {"train_loss": 0.8})
    tracker.log_summary({"final_loss": 0.1})
    tracker.close()

    run_dir = tmp_path / "test_run_01"
    assert run_dir.exists()
    meta_path = run_dir / "run.json"
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_id"] == "test_run_01"
    assert meta["params"]["model"] == "resnet18"
    assert meta["summary"]["final_loss"] == 0.1
    assert meta["status"] == "completed"


def test_keras_callback():
    cb = AutotrainerKerasCallback(log_batches=True)
    cb.on_train_batch_end(0, {"loss": 0.99})
    cb.on_epoch_end(0, {"loss": 0.5, "val_loss": 0.4, "val_accuracy": 0.85})
    assert cb.step_count == 1


def test_xgboost_callback():
    cb = autotrainer_xgboost_callback()

    class MockEnv:
        evaluation_result_list = [("train-rmse", 0.5), ("val-rmse", 0.4)]

    cb(MockEnv())


def test_lightgbm_callback():
    cb = autotrainer_lightgbm_callback()

    class MockEnv:
        evaluation_result_list = [("train", "rmse", 0.5, False), ("val", "rmse", 0.4, False)]

    cb(MockEnv())
