"""Tests for doctor.py: the pre-flight environment diagnostic."""

from __future__ import annotations

import socket

from autotrainer.doctor import (
    WARN,
    _check_cpus,
    _check_port,
    _check_scratch,
    _check_slurm,
    run_doctor,
)


class TestCheckSlurm:
    def test_flags_ntasks_gpus_mismatch(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "42")
        monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "4")
        monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
        report: list[str] = []
        _check_slurm(report)
        assert any("ntasks-per-node=4" in line and "gpus-on-node=2" in line for line in report)

    def test_no_warning_when_matching(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "42")
        monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "2")
        monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
        report: list[str] = []
        _check_slurm(report)
        # No mismatch line should appear.
        assert not any("ntasks-per-node" in line for line in report)

    def test_local_mode_is_ok(self):
        # clean_env fixture ensures no SLURM_JOB_ID.
        report: list[str] = []
        _check_slurm(report)
        assert any("not inside a SLURM job" in line for line in report)


class TestCheckCpus:
    """The loader-worker budget is what actually starves a GPU, so doctor
    should report it before a job spends the allocation finding out."""

    def test_reports_the_worker_count_it_will_use(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        report: list[str] = []
        _check_cpus(report)
        assert any("num_workers=8" in line and "SLURM allocation" in line for line in report)

    def test_warns_when_a_single_cpu_must_feed_a_gpu(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
        report: list[str] = []
        _check_cpus(report)
        assert any(line.startswith(WARN) and "cannot keep a GPU fed" in line for line in report)

    def test_caps_the_reported_workers(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "128")
        report: list[str] = []
        _check_cpus(report)
        assert any("num_workers=8" in line for line in report)

    def test_off_slurm_names_the_affinity_mask(self, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        report: list[str] = []
        _check_cpus(report)
        assert any("affinity mask" in line for line in report)

    def test_missing_cpus_per_task_is_flagged_inside_a_job(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "42")
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        report: list[str] = []
        _check_slurm(report)
        assert any("SLURM_CPUS_PER_TASK unset" in line for line in report)


class TestCheckScratch:
    def test_node_local_scratch_is_ok(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        report: list[str] = []
        _check_scratch(report)
        assert any("looks node-local" in line for line in report)

    def test_networked_scratch_is_flagged(self, monkeypatch):
        import autotrainer.slurm as slurm_mod

        monkeypatch.setattr(slurm_mod, "_looks_networked", lambda p: True)
        report: list[str] = []
        _check_scratch(report)
        assert any(line.startswith(WARN) and "looks networked" in line for line in report)


class TestCheckPort:
    def test_warns_when_port_in_use(self, monkeypatch):
        # Bind a socket on a free port, then ask doctor to check that port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.listen(1)
            monkeypatch.setenv("AUTOTRAINER_PORT", str(port))
            report: list[str] = []
            _check_port(report)
            assert any(line.startswith(WARN) and str(port) in line for line in report)
            # Keep the socket alive until the assertion is done.

    def test_free_port_is_ok(self, monkeypatch):
        # Pick an ephemeral port we never bind, so it should be free.
        monkeypatch.setenv("AUTOTRAINER_PORT", "49999")
        report: list[str] = []
        _check_port(report)
        assert any("is free" in line for line in report)


class TestRunDoctor:
    def test_returns_zero_on_healthy_env(self, monkeypatch):
        # A clean env with at least one framework installed is healthy.
        monkeypatch.setenv("AUTOTRAINER_PORT", "49998")
        assert run_doctor() == 0
