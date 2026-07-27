"""Tests for triage.py: the TrainingMonitor numerical-health diagnostics.

Each detector is exercised in isolation with synthetic signals (plain-float
losses, a fake AMP scaler, hand-set ``.grad`` tensors) so the whole file runs
on CPU with no training loop. Diagnostics are asserted via the monitor's
``diagnostics`` list; the fire-once and all-clear behaviors are covered too.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from autotrainer.triage import TrainingMonitor  # noqa: E402


class _FakeScaler:
    """Minimal GradScaler stand-in: a scripted sequence of scales."""

    def __init__(self, scales, enabled=True):
        self._scales = list(scales)
        self._i = 0
        self._enabled = enabled

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        s = self._scales[min(self._i, len(self._scales) - 1)]
        self._i += 1
        return s


class _FakeOpt:
    def __init__(self, lr):
        self.param_groups = [{"lr": lr}]


def _set_grad(model, value):
    for p in model.parameters():
        p.grad = torch.full_like(p, float(value))


def _joined(mon):
    return " ".join(mon.diagnostics)


class TestLossDetectors:
    def test_nan_loss_flagged_immediately(self):
        mon = TrainingMonitor()
        mon.step(float("nan"))
        assert any("loss is NaN" in m for m in mon.diagnostics)

    def test_inf_loss_flagged(self):
        mon = TrainingMonitor()
        mon.step(float("inf"))
        assert any("loss is Inf" in m for m in mon.diagnostics)

    def test_nan_loss_includes_concrete_lr_hint_when_optimizer_given(self):
        mon = TrainingMonitor()
        mon.step(float("nan"), optimizer=_FakeOpt(0.1))
        assert "lr=" in _joined(mon)  # 0.1 / 10 -> a concrete suggestion

    def test_tensor_loss_is_accepted(self):
        mon = TrainingMonitor()
        mon.step(torch.tensor(float("nan")))
        assert any("loss is NaN" in m for m in mon.diagnostics)

    def test_diverging_loss_flagged(self):
        mon = TrainingMonitor(warmup=3)
        for _ in range(5):
            mon.step(1.0)
        for _ in range(8):
            mon.step(10.0)
        assert any("diverging" in m for m in mon.diagnostics)

    def test_plateau_flagged(self):
        # High divergence_factor so the flat loss takes the plateau path.
        mon = TrainingMonitor(warmup=2, plateau_patience=5, divergence_factor=100)
        for _ in range(12):
            mon.step(1.0)
        assert any("hasn't improved" in m for m in mon.diagnostics)

    def test_healthy_decreasing_loss_is_all_clear(self):
        mon = TrainingMonitor(warmup=2)
        for v in [5.0, 4.0, 3.0, 2.5, 2.0, 1.7, 1.5, 1.3, 1.2, 1.1]:
            mon.step(v)
        assert mon.diagnostics == []
        assert "no issues detected" in mon.report()


class TestScalerOverflow:
    def test_persistent_backoff_flagged(self):
        mon = TrainingMonitor()  # overflow_patience=3
        scaler = _FakeScaler([65536, 32768, 16384, 8192, 4096])
        for _ in range(5):
            mon.step(0.5, scaler=scaler)
        assert any("fp16 gradient overflow" in m for m in mon.diagnostics)

    def test_stable_scale_is_not_flagged(self):
        mon = TrainingMonitor()
        scaler = _FakeScaler([65536] * 6)
        for _ in range(6):
            mon.step(0.5, scaler=scaler)
        assert "overflow" not in _joined(mon)

    def test_disabled_scaler_is_ignored(self):
        mon = TrainingMonitor()
        scaler = _FakeScaler([65536, 32768, 16384, 8192, 4096], enabled=False)
        for _ in range(5):
            mon.step(0.5, scaler=scaler)
        assert "overflow" not in _joined(mon)


class TestGradDetectors:
    def test_non_finite_grad_flagged(self):
        model = nn.Linear(3, 2)
        _set_grad(model, float("nan"))
        mon = TrainingMonitor()
        mon.step(0.5, model=model)
        assert any("gradients are NaN" in m for m in mon.diagnostics)

    def test_grad_spike_flagged(self):
        model = nn.Linear(3, 2)
        mon = TrainingMonitor(warmup=2, spike_factor=10)
        for _ in range(5):
            _set_grad(model, 0.1)
            mon.step(0.5, model=model)
        _set_grad(model, 100.0)  # ~1000x the recent norm
        mon.step(0.5, model=model)
        assert any("spiked" in m for m in mon.diagnostics)

    def test_vanishing_grad_flagged(self):
        model = nn.Linear(3, 2)
        mon = TrainingMonitor(warmup=2)
        for _ in range(4):
            _set_grad(model, 1e-10)
            mon.step(0.5, model=model)
        assert any("vanishing" in m for m in mon.diagnostics)

    def test_grad_checks_skipped_while_scaler_active(self):
        # Grads are still scaled when an AMP scaler is active, so their
        # magnitude is meaningless - the monitor must not read them.
        model = nn.Linear(3, 2)
        _set_grad(model, float("nan"))
        mon = TrainingMonitor()
        mon.step(0.5, model=model, scaler=_FakeScaler([65536]))
        assert "gradients are" not in _joined(mon)

    def test_no_grads_yet_is_safe(self):
        mon = TrainingMonitor()
        mon.step(0.5, model=nn.Linear(3, 2))  # params have no .grad
        assert mon.diagnostics == []


class TestFireOnceAndReport:
    def test_each_issue_fires_at_most_once(self):
        mon = TrainingMonitor()
        for _ in range(5):
            mon.step(float("nan"))
        assert sum("loss is NaN" in m for m in mon.diagnostics) == 1

    def test_report_recaps_when_issues_fired(self):
        mon = TrainingMonitor()
        mon.step(float("nan"))
        assert "issue(s) flagged" in mon.report()

    def test_report_is_none_before_any_step(self):
        assert TrainingMonitor().report() is None


class TestContract:
    def test_step_with_only_loss_needs_nothing_else(self):
        mon = TrainingMonitor()
        mon.step(0.5)  # no model / optimizer / scaler
        assert mon.diagnostics == []

    def test_does_not_mutate_model_or_optimizer(self):
        model = nn.Linear(3, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        _set_grad(model, 0.5)
        mon = TrainingMonitor()
        for _ in range(3):
            mon.step(0.5, model=model, optimizer=opt)
        assert opt.param_groups[0]["lr"] == 0.1  # lr untouched (observe only)
        for k, v in model.state_dict().items():
            assert torch.equal(before[k], v)  # weights untouched

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"warmup": -1},
            {"window": 0},
            {"divergence_factor": 1.0},
            {"spike_factor": 1.0},
            {"plateau_patience": 0},
            {"overflow_patience": 0},
        ],
    )
    def test_invalid_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            TrainingMonitor(**kwargs)
