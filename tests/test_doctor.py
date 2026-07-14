"""Tests for doctor.py: the pre-flight environment diagnostic."""

from __future__ import annotations

import socket

from autotrainer.doctor import WARN, _check_port, _check_slurm, run_doctor


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
