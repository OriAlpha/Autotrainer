"""Zero-dependency native experiment tracker for autotrainer.

Writes structured metrics (CSV, JSONL, run metadata JSON) to local disk
under ``./logs/<run_id>/`` without requiring third-party libraries.
Powers the native ``autotrainer ui`` web dashboard.
"""

from __future__ import annotations

import csv
import getpass
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


def _get_current_user(user: str | None = None) -> str:
    """Detect current OS / cluster user identity."""
    if user:
        return user.strip()
    for env_key in ("AUTOTRAINER_USER", "SLURM_JOB_USER", "USER", "USERNAME", "LOGNAME"):
        val = os.getenv(env_key)
        if val:
            return val.strip()
    try:
        return getpass.getuser().strip()
    except Exception:
        return "default"


@runtime_checkable
class BaseTracker(Protocol):
    """Protocol for experiment trackers in autotrainer."""

    def log_params(self, params: dict[str, Any]) -> None:
        ...

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        ...

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        ...

    def log_summary(self, summary: dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...


class CSVTracker:
    """Standard library CSV metric logger."""

    DEFAULT_FIELDS = ["timestamp", "epoch", "step", "train_loss", "val_loss", "val_acc", "loss"]

    def __init__(self, log_dir: str | Path, filename: str = "metrics.csv") -> None:
        self.log_dir = Path(log_dir)
        self.filepath = self.log_dir / filename
        self.fieldnames: list[str] = list(self.DEFAULT_FIELDS)
        self._file = None
        self._writer = None

    def _ensure_writer(self, sample_dict: dict[str, Any]) -> None:
        if self._writer is None:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.filepath.exists() and self.filepath.stat().st_size > 0
            for k in sample_dict.keys():
                if k not in self.fieldnames:
                    self.fieldnames.append(k)
            
            # Open file for appending
            self._file = open(self.filepath, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            if not file_exists:
                self._writer.writeheader()
                self._file.flush()

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        row: dict[str, Any] = {"timestamp": round(time.time(), 3), "epoch": epoch}
        row.update(metrics)
        self._ensure_writer(row)
        if self._writer and self._file:
            filtered_row = {k: row.get(k, "") for k in self.fieldnames}
            self._writer.writerow(filtered_row)
            self._file.flush()

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        row: dict[str, Any] = {"timestamp": round(time.time(), 3), "step": step}
        row.update(metrics)
        self._ensure_writer(row)
        if self._writer and self._file:
            filtered_row = {k: row.get(k, "") for k in self.fieldnames}
            self._writer.writerow(filtered_row)
            self._file.flush()

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_summary(self, summary: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()


class JSONLTracker:
    """Standard library JSON Lines logger."""

    def __init__(self, log_dir: str | Path, filename: str = "metrics.jsonl") -> None:
        self.log_dir = Path(log_dir)
        self.filepath = self.log_dir / filename
        self._file = None

    def _get_file(self):
        if self._file is None or self._file.closed:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.filepath, "a", encoding="utf-8")
        return self._file

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        record = {
            "type": "epoch",
            "timestamp": round(time.time(), 3),
            "epoch": epoch,
            "metrics": metrics,
        }
        f = self._get_file()
        f.write(json.dumps(record) + "\n")
        f.flush()

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        record = {
            "type": "step",
            "timestamp": round(time.time(), 3),
            "step": step,
            "metrics": metrics,
        }
        f = self._get_file()
        f.write(json.dumps(record) + "\n")
        f.flush()

    def log_params(self, params: dict[str, Any]) -> None:
        record = {
            "type": "params",
            "timestamp": round(time.time(), 3),
            "params": params,
        }
        f = self._get_file()
        f.write(json.dumps(record) + "\n")
        f.flush()

    def log_summary(self, summary: dict[str, Any]) -> None:
        record = {
            "type": "summary",
            "timestamp": round(time.time(), 3),
            "summary": summary,
        }
        f = self._get_file()
        f.write(json.dumps(record) + "\n")
        f.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()


class NativeTracker:
    """Native autotrainer experiment tracker.
    
    Creates a unique run folder under ``./logs/<run_id>`` and writes CSV,
    JSONL, and run metadata JSON files for rendering in ``autotrainer ui``.
    """

    def __init__(
        self,
        base_dir: str | Path = "logs",
        run_name: str | None = None,
        user: str | None = None,
        logs_dir: str | Path | None = None,
    ) -> None:
        target_dir = logs_dir if logs_dir is not None else base_dir
        self.base_dir = Path(target_dir)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = run_name or f"run_{timestamp_str}"
        self.user = _get_current_user(user)
        self.run_dir = self.base_dir / self.run_id
        self._initialized = False

        self.csv_tracker = CSVTracker(self.run_dir)
        self.jsonl_tracker = JSONLTracker(self.run_dir)

        self.metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "user": self.user,
            "start_time": time.time(),
            "start_datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": {},
            "summary": {},
            "status": "running",
            "tags": [],
            "favorite": False,
            "notes": "",
            "archived": False,
            "hardware": self._detect_hardware(),
        }

    def _detect_hardware(self) -> dict[str, Any]:
        """Detect available GPU and host hardware metrics."""
        hw: dict[str, Any] = {}
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                hw["gpu_name"] = props.name
                hw["gpu_total_mb"] = round(props.total_memory / (1024 * 1024), 1)
                hw["gpu_count"] = torch.cuda.device_count()
        except Exception:
            pass

        try:
            import psutil
            hw["cpu_count"] = psutil.cpu_count(logical=True)
            hw["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            pass
        return hw

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            self._save_metadata()

    def _save_metadata(self) -> None:
        if self._initialized:
            meta_path = self.run_dir / "run.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)

    def add_tag(self, tag: str) -> None:
        self._ensure_init()
        clean_tag = tag.strip().lower()
        if clean_tag and clean_tag not in self.metadata["tags"]:
            self.metadata["tags"].append(clean_tag)
            self._save_metadata()

    def remove_tag(self, tag: str) -> None:
        self._ensure_init()
        clean_tag = tag.strip().lower()
        if clean_tag in self.metadata["tags"]:
            self.metadata["tags"].remove(clean_tag)
            self._save_metadata()

    def set_favorite(self, is_favorite: bool) -> None:
        self._ensure_init()
        self.metadata["favorite"] = bool(is_favorite)
        self._save_metadata()

    def set_notes(self, notes: str) -> None:
        self._ensure_init()
        self.metadata["notes"] = str(notes)
        self._save_metadata()
        try:
            (self.run_dir / "notes.md").write_text(str(notes), encoding="utf-8")
        except Exception:
            pass

    def set_archived(self, is_archived: bool) -> None:
        self._ensure_init()
        self.metadata["archived"] = bool(is_archived)
        self._save_metadata()

    def log_params(self, params: dict[str, Any]) -> None:
        self._ensure_init()
        self.metadata["params"].update(params)
        self._save_metadata()
        self.jsonl_tracker.log_params(params)

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        self._ensure_init()
        self.csv_tracker.log_step(step, metrics)
        self.jsonl_tracker.log_step(step, metrics)

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        self._ensure_init()
        self.csv_tracker.log_epoch(epoch, metrics)
        self.jsonl_tracker.log_epoch(epoch, metrics)

    def log_summary(self, summary: dict[str, Any]) -> None:
        self._ensure_init()
        self.metadata["summary"].update(summary)
        self.metadata["status"] = "completed"
        self.metadata["end_time"] = time.time()
        self.metadata["end_datetime"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_metadata()
        self.jsonl_tracker.log_summary(summary)

    def close(self) -> None:
        self.csv_tracker.close()
        self.jsonl_tracker.close()
