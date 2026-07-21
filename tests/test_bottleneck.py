"""Tests for the BottleneckMonitor - the cheapest piece of training triage."""

from __future__ import annotations

import time

import pytest

from autotrainer.bottleneck import BottleneckMonitor


class TestStatsBeforeWarmup:
    def test_returns_zeros_until_warmup(self):
        mon = BottleneckMonitor(warmup=5, window=10)
        # Record a couple of samples but not enough to clear warmup.
        with mon.data_time():
            time.sleep(0.001)
        with mon.step_time():
            time.sleep(0.001)
        mon.tick()
        s = mon.stats()
        assert s["data_fraction"] == 0.0


class TestStatsAfterWarmup:
    def test_records_and_computes_fraction(self):
        mon = BottleneckMonitor(warmup=1, window=10)
        # data_time dominates step_time -> fraction should be high.
        for _ in range(3):
            with mon.data_time():
                time.sleep(0.01)
            with mon.step_time():
                time.sleep(0.001)
            mon.tick()
        s = mon.stats()
        assert s["data_time"] > 0
        assert s["step_time"] > 0
        assert s["data_fraction"] > 0.6  # data dominated


class TestReport:
    def test_should_report_false_until_warmup(self):
        mon = BottleneckMonitor(warmup=10, window=5, report_every=1)
        for _ in range(5):
            with mon.data_time():
                pass
            with mon.step_time():
                pass
            mon.tick()
        # Not enough ticks yet (< warmup=10).
        assert mon.should_report() is False

    def test_report_returns_message_after_warmup(self, capsys):
        mon = BottleneckMonitor(warmup=2, window=10, report_every=1)
        for _ in range(5):
            with mon.data_time():
                time.sleep(0.01)  # dominate
            with mon.step_time():
                time.sleep(0.001)
            mon.tick()
        assert mon.should_report()
        msg = mon.report()
        assert msg is not None
        assert "bottleneck" in msg
        out = capsys.readouterr().out
        assert "bottleneck" in out  # printed via print0

    def test_report_skips_when_insufficient_data(self):
        mon = BottleneckMonitor(warmup=5, window=10)
        # No samples recorded.
        assert mon.report() is None

    def test_report_every_throttles(self):
        mon = BottleneckMonitor(warmup=1, window=5, report_every=100)
        for _ in range(5):
            with mon.data_time():
                time.sleep(0.001)
            with mon.step_time():
                time.sleep(0.001)
            mon.tick()
        # First report fires (ticks >= warmup, and last_report was -100).
        assert mon.should_report()
        mon.report()
        # Immediately after, should_report is False until report_every ticks pass.
        for _ in range(5):
            with mon.data_time():
                pass
            with mon.step_time():
                pass
            mon.tick()
        assert mon.should_report() is False


class TestValidation:
    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            BottleneckMonitor(warmup=-1)
        with pytest.raises(ValueError):
            BottleneckMonitor(window=0)
        with pytest.raises(ValueError):
            BottleneckMonitor(report_every=0)


class TestReportClassification:
    def _record(self, mon, data_s, step_s, n=5):
        """Record fake timings without actually sleeping (fast + deterministic)."""
        for _ in range(n):
            mon._data_times.append(data_s)
            mon._step_times.append(step_s)
            mon._ticks += 1

    def test_gpu_bound_message(self, capsys):
        mon = BottleneckMonitor(warmup=1, window=10, report_every=1)
        self._record(mon, data_s=0.001, step_s=0.05)  # 2% data -> GPU bound
        msg = mon.report()
        assert "GPU-bound" in msg

    def test_bottleneck_message(self, capsys):
        mon = BottleneckMonitor(warmup=1, window=10, report_every=1)
        self._record(mon, data_s=0.05, step_s=0.001)  # 98% data -> bottleneck
        msg = mon.report()
        assert "dataloader is" in msg and "raise num_workers" in msg

    def test_balanced_message(self, capsys):
        mon = BottleneckMonitor(warmup=1, window=10, report_every=1)
        self._record(mon, data_s=0.01, step_s=0.01)  # 50% data -> neutral
        msg = mon.report()
        # Neutral case: neither GPU-bound nor "raise num_workers".
        assert "GPU-bound" not in msg
        assert "raise num_workers" not in msg
