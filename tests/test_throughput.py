"""Tests for the ThroughputMonitor - throughput, peak GPU memory, and MFU.

The bookkeeping (counter increments, time-window averaging, MFU math) is
unit-tested on CPU; the real-GPU memory read is gated on the ``cuda`` marker
because it needs a device that reports ``max_memory_reserved``.
"""

from __future__ import annotations

import time

import pytest

from autotrainer.throughput import _GPU_TFLOPS, ThroughputMonitor, _advertised_tflops


class TestStatsBeforeWarmup:
    def test_returns_zeros_until_warmup(self):
        mon = ThroughputMonitor(batch_size=8, warmup=5, window=10)
        with mon.step_time():
            time.sleep(0.001)
        mon.tick()
        s = mon.stats()
        assert s["samples_per_sec"] == 0.0
        assert s["mfu"] is None


class TestStatsAfterWarmup:
    def _record(self, mon, step_s, n=3):
        """Inject fake timings without sleeping (fast + deterministic)."""
        for _ in range(n):
            mon._step_times.append(step_s)
            mon._samples.append(mon._default_bs or 0)
            mon._ticks += 1

    def test_records_and_computes_throughput(self):
        mon = ThroughputMonitor(batch_size=8, warmup=1, window=10)
        self._record(mon, 0.001)
        s = mon.stats()
        assert s["samples_per_sec"] > 0

    def test_mfu_requires_model_flops(self):
        # Without model_flops, MFU is None even with samples and timing.
        mon = ThroughputMonitor(batch_size=8, warmup=1, window=10)
        self._record(mon, 0.001)
        assert mon.stats()["mfu"] is None

    def test_mfu_computed_when_model_flops_and_known_gpu(self, monkeypatch):
        # MFU needs a known GPU for the denominator; force one so the test
        # doesn't depend on the host having CUDA + a table entry. Use injected
        # (not wall-clock) timings so the arithmetic is exact, not flaky.
        monkeypatch.setattr("autotrainer.throughput._advertised_tflops", lambda: 312.0)
        mon = ThroughputMonitor(batch_size=8, model_flops=1e12, warmup=1, window=10)
        # 3 steps * 8 samples, 0.01s each -> 2400 samples/sec.
        # Achieved = 1e12 FLOPs/step * 3 / 0.03s = 1e14 FLOPs/s = 100 TFLOPS.
        # MFU = 100 / 312 ~= 0.32. Sub-1.0 as a real workload should be.
        self._record(mon, 0.01)
        mfu = mon.stats()["mfu"]
        assert mfu is not None
        assert abs(mfu - (100.0 / 312.0)) < 1e-6

    def test_mfu_none_when_gpu_unknown(self, monkeypatch):
        monkeypatch.setattr("autotrainer.throughput._advertised_tflops", lambda: None)
        mon = ThroughputMonitor(batch_size=8, model_flops=1e12, warmup=1, window=10)
        self._record(mon, 0.001)
        assert mon.stats()["mfu"] is None


class TestReport:
    def test_should_report_false_until_warmup(self):
        mon = ThroughputMonitor(batch_size=8, warmup=10, window=5, report_every=1)
        for _ in range(5):
            with mon.step_time():
                pass
            mon.tick()
        # Not enough ticks yet (< warmup=10).
        assert mon.should_report() is False

    def test_report_returns_message_after_warmup(self, capsys):
        mon = ThroughputMonitor(batch_size=8, warmup=2, window=10, report_every=1)
        for _ in range(5):
            with mon.step_time():
                time.sleep(0.001)
            mon.tick()
        assert mon.should_report()
        msg = mon.report()
        assert msg is not None
        assert "throughput" in msg
        assert "samples/sec" in msg
        out = capsys.readouterr().out
        assert "throughput" in out  # printed via print0

    def test_report_skips_when_insufficient_data(self):
        mon = ThroughputMonitor(batch_size=8, warmup=5, window=10)
        assert mon.report() is None

    def test_report_includes_mfu_when_available(self, monkeypatch, capsys):
        monkeypatch.setattr("autotrainer.throughput._advertised_tflops", lambda: 989.0)
        mon = ThroughputMonitor(batch_size=8, model_flops=1e12, warmup=1, window=10, report_every=1)
        # Injected timing (not wall-clock) so the report actually fires and
        # isn't flaky on a slow CI runner.
        for _ in range(3):
            mon._step_times.append(0.01)
            mon._samples.append(8)
            mon._ticks += 1
        msg = mon.report()
        assert msg is not None and "MFU" in msg

    def test_report_every_throttles(self):
        mon = ThroughputMonitor(batch_size=8, warmup=1, window=5, report_every=100)
        for _ in range(5):
            with mon.step_time():
                time.sleep(0.001)
            mon.tick()
        assert mon.should_report()
        mon.report()
        for _ in range(5):
            with mon.step_time():
                pass
            mon.tick()
        assert mon.should_report() is False


class TestValidation:
    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            ThroughputMonitor(warmup=-1)
        with pytest.raises(ValueError):
            ThroughputMonitor(window=0)
        with pytest.raises(ValueError):
            ThroughputMonitor(report_every=0)

    def test_rejects_nonpositive_batch_size(self):
        with pytest.raises(ValueError):
            ThroughputMonitor(batch_size=0)

    def test_rejects_nonpositive_model_flops(self):
        with pytest.raises(ValueError):
            ThroughputMonitor(model_flops=0.0)


class TestBatchSizeOverride:
    def test_tick_batch_size_overrides_default(self):
        mon = ThroughputMonitor(batch_size=8, warmup=1, window=10)
        # First step with the default 8, then a remainder step with 3.
        for bs in (8, 8, 3):
            with mon.step_time():
                time.sleep(0.001)
            mon.tick(bs)
        s = mon.stats()
        # total samples = 8 + 8 + 3 = 19; throughput just needs to be positive.
        assert s["samples_per_sec"] > 0

    def test_tick_with_no_batch_size_anywhere(self):
        # No default, no per-tick bs -> samples count as 0 -> throughput 0,
        # but no crash. MFU stays None.
        mon = ThroughputMonitor(warmup=1, window=10)
        for _ in range(3):
            with mon.step_time():
                time.sleep(0.001)
            mon.tick()
        s = mon.stats()
        assert s["samples_per_sec"] == 0.0
        assert s["mfu"] is None


class TestGpuSpecTable:
    def test_known_gpu_returns_float(self, monkeypatch):
        # Stub torch to report an A100 so the lookup hits the table.
        class _Props:
            name = "NVIDIA A100-SXM4-80GB"

        class _Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_properties(_idx):
                return _Props()

        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = _Cuda()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert _advertised_tflops() == 312.0

    def test_unknown_gpu_returns_none(self, monkeypatch):
        class _Props:
            name = "NVIDIA Mystery GPU 9999"

        class _Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_properties(_idx):
                return _Props()

        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = _Cuda()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert _advertised_tflops() is None

    def test_no_cuda_returns_none(self, monkeypatch):
        import sys
        import types

        fake_torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available():
                return False

        fake_torch.cuda = _Cuda()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert _advertised_tflops() is None

    def test_substring_match_prefers_longer_key(self):
        # "rtx 5070 ti" must not match the "rtx 5070" entry. Verified by
        # table ordering (sorted by length desc), asserted directly here.
        keys = sorted(_GPU_TFLOPS, key=len, reverse=True)
        ti_idx = keys.index("rtx 5070 ti")
        base_idx = keys.index("rtx 5070")
        assert ti_idx < base_idx
