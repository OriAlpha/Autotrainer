"""Tests for ``autotrainer.train_step`` - the all-in-one AMP training step.

Two layers:

  * CPU tests that actually run forward -> loss -> backward -> step -> zero
    through the helper and assert the model *learns* (loss drops, params
    update). This is the end-to-end coverage the AMP path never had before:
    the individual pieces (autocast_context, GradScaler) were unit-tested,
    but the full step the docs tell users to run was never executed.
  * A ``cuda``-marked test that runs the same loop with a *real* fp16
    GradScaler on a GPU, so the actual scaling path is exercised on the
    self-hosted runner (on CPU the scaler is disabled and autocast is a
    nullcontext, so only the plumbing is covered there).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


def _regression_problem(seed: int = 0):
    """A tiny, deterministic linear-regression task a Linear must be able to
    fit - full-batch GD on it makes the loss decrease monotonically, so
    'did the step actually train?' is a stable assertion."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    x = torch.randn(64, 4)
    w_true = torch.randn(4, 1)
    y = x @ w_true + 0.5
    return model, opt, loss_fn, x, y


class TestTrainStepCpuEndToEnd:
    def test_loss_decreases_over_steps(self):
        """The end-to-end contract: repeated train_step calls train the model."""
        from autotrainer import train_step

        model, opt, loss_fn, x, y = _regression_problem()
        first = None
        last = None
        for _ in range(200):
            loss = train_step(model, loss_fn, x, y, opt)  # scaler=None (CPU)
            if first is None:
                first = loss.item()
            last = loss.item()
        assert last < first * 0.5, f"loss barely moved: {first:.4f} -> {last:.4f}"

    def test_returns_detached_scalar_loss(self):
        import torch

        from autotrainer import train_step

        model, opt, loss_fn, x, y = _regression_problem()
        loss = train_step(model, loss_fn, x, y, opt)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad is False  # detached: safe to log/retain
        assert loss.ndim == 0

    def test_grads_are_zeroed_after_step(self):
        """zero_grad(set_to_none=True) runs last, so grads are None afterwards -
        the next step starts clean without the user calling zero_grad."""
        from autotrainer import train_step

        model, opt, loss_fn, x, y = _regression_problem()
        train_step(model, loss_fn, x, y, opt)
        assert all(p.grad is None for p in model.parameters())

    def test_does_not_touch_the_recipe(self):
        """lr / optimizer state ownership stays with the user."""
        from autotrainer import train_step

        model, opt, loss_fn, x, y = _regression_problem()
        lr_before = [g["lr"] for g in opt.param_groups]
        train_step(model, loss_fn, x, y, opt)
        assert [g["lr"] for g in opt.param_groups] == lr_before

    def test_disabled_scaler_is_accepted_and_trains(self):
        """Passing GradScaler() (disabled on CPU) must take the scaler path and
        still train - the same code the user copies runs unchanged on CPU."""
        from autotrainer import GradScaler, train_step

        model, opt, loss_fn, x, y = _regression_problem()
        scaler = GradScaler()
        assert scaler.is_enabled() is False  # CPU: disabled, pass-through
        first = None
        last = None
        for _ in range(200):
            loss = train_step(model, loss_fn, x, y, opt, scaler=scaler)
            if first is None:
                first = loss.item()
            last = loss.item()
        assert last < first * 0.5

    def test_autocast_false_still_trains(self):
        from autotrainer import train_step

        model, opt, loss_fn, x, y = _regression_problem()
        first = None
        last = None
        for _ in range(200):
            loss = train_step(model, loss_fn, x, y, opt, autocast=False)
            if first is None:
                first = loss.item()
            last = loss.item()
        assert last < first * 0.5

    def test_dict_inputs_dispatch_through_robust_forward(self):
        """A dict input is called as model(**inputs), matching auto()/fit()."""
        import torch.nn as nn

        from autotrainer import train_step

        class DictModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 1)

            def forward(self, feats):
                return self.lin(feats)

        import torch

        torch.manual_seed(0)
        model = DictModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        x = torch.randn(64, 4)
        y = torch.randn(64, 1)
        loss = train_step(model, loss_fn, {"feats": x}, y, opt)
        assert loss.requires_grad is False


@pytest.mark.cuda
@pytest.mark.skipif(not _has_cuda(), reason="real fp16 AMP scaling needs a CUDA GPU")
class TestTrainStepCudaAmp:
    """The real-hardware AMP end-to-end. On an fp16 GPU the GradScaler is
    enabled, so this exercises actual loss scaling; on a bf16 GPU the scaler
    is disabled but autocast still runs in bf16 - both are genuine on-device
    AMP steps the CPU tests can't cover."""

    def test_amp_step_trains_and_stays_finite(self):
        import torch
        import torch.nn as nn

        from autotrainer import GradScaler, train_step

        torch.manual_seed(0)
        device = torch.device("cuda:0")
        model = nn.Linear(16, 4).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        loss_fn = nn.MSELoss()
        x = torch.randn(128, 16, device=device)
        w_true = torch.randn(16, 4, device=device)
        y = x @ w_true
        scaler = GradScaler()

        first = None
        last = None
        for _ in range(100):
            loss = train_step(model, loss_fn, x, y, opt, scaler=scaler)
            assert torch.isfinite(loss).all(), "AMP produced a non-finite loss"
            if first is None:
                first = loss.item()
            last = loss.item()
        assert last < first, f"AMP step did not train: {first:.4f} -> {last:.4f}"
        assert torch.isfinite(model.weight).all()

    def test_enabled_fp16_scaler_scaling_path_trains(self):
        """Force an enabled GradScaler so the real scale -> unscale -> step ->
        update arithmetic runs even on a bf16 GPU (where autotrainer's own
        GradScaler() is disabled). This covers the fp16 scaling path on ANY
        CUDA device, not just fp16-only hardware the runner may not have."""
        import torch
        import torch.nn as nn

        from autotrainer import train_step

        torch.manual_seed(0)
        device = torch.device("cuda:0")
        model = nn.Linear(16, 4).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        loss_fn = nn.MSELoss()
        x = torch.randn(128, 16, device=device)
        y = x @ torch.randn(16, 4, device=device)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        assert scaler.is_enabled()

        first = None
        last = None
        for _ in range(100):
            loss = train_step(model, loss_fn, x, y, opt, scaler=scaler)
            assert torch.isfinite(loss).all()
            if first is None:
                first = loss.item()
            last = loss.item()
        assert last < first, f"scaled AMP step did not train: {first:.4f} -> {last:.4f}"
        assert scaler.get_scale() > 0  # scaler stayed healthy (not stuck at inf)

    def test_amp_step_accepts_batch_not_pre_moved_to_device(self):
        """train_step moves inputs to the model's device, so a CPU batch against
        a CUDA model must just work (the common 'forgot .to(device)' case)."""
        import torch
        import torch.nn as nn

        from autotrainer import GradScaler, train_step

        torch.manual_seed(0)
        model = nn.Linear(8, 2).to("cuda:0")
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        loss_fn = nn.MSELoss()
        x = torch.randn(32, 8)  # left on CPU on purpose
        y = torch.randn(32, 2)
        loss = train_step(model, loss_fn, x, y, opt, scaler=GradScaler())
        assert torch.isfinite(loss).all()
