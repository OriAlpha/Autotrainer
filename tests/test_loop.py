"""Tests for the training-loop helpers in loop.py.

Contract: these helpers never touch lr / loss / schedule / optimizer choice.
They are pure ergonomics - zero_grad, eval/train mode guards, and grad
accumulation. The optimizer passed in is the optimizer passed out.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


class TestZeroGrad:
    def test_clears_existing_grads(self):
        opt = torch.optim.SGD(nn.Linear(3, 2).parameters(), lr=0.1)
        # Plant a grad so zero_grad has something to clear.
        for p in opt.param_groups[0]["params"]:
            p.grad = torch.ones_like(p)
        from autotrainer.loop import zero_grad

        zero_grad(opt)
        assert all(p.grad is None for p in opt.param_groups[0]["params"])


class TestEvalMode:
    def test_sets_eval_then_restores_train(self):
        from autotrainer.loop import eval_mode

        model = nn.Linear(3, 2)
        model.train()
        assert model.training
        with eval_mode(model):
            assert not model.training
        assert model.training  # restored

    def test_restores_eval_state_if_started_eval(self):
        from autotrainer.loop import eval_mode

        model = nn.Linear(3, 2)
        model.eval()
        with eval_mode(model):
            assert not model.training
        assert not model.training  # was eval before, still eval after

    def test_non_module_passes_through(self):
        from autotrainer.loop import eval_mode

        # A bare object with no .train/.eval should just pass through.
        obj = object()
        with eval_mode(obj) as inner:
            assert inner is obj


class TestTrainMode:
    def test_sets_train_then_restores_eval(self):
        from autotrainer.loop import train_mode

        model = nn.Linear(3, 2)
        model.eval()
        with train_mode(model):
            assert model.training
        assert not model.training


class TestAccumulate:
    def _setup(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        # Save initial params so we can confirm stepping actually moved them.
        init = {k: v.clone() for k, v in model.state_dict().items()}
        return model, opt, loss_fn, init

    def test_steps_once_per_block_with_steps_1(self):
        from autotrainer.loop import accumulate

        model, opt, loss_fn, init = self._setup()
        with accumulate(opt, steps=1) as acc:
            for _ in range(3):
                x = torch.randn(8, 4)
                y = torch.randn(8, 2)
                acc.backward(loss_fn(model(x), y))
        # steps=1 means each backward triggers an optimizer step; grads cleared
        # after each, so params did move.
        moved = any(not torch.equal(model.state_dict()[k], init[k]) for k in init)
        assert moved

    def test_accumulates_then_steps_with_steps_4(self):
        from autotrainer.loop import accumulate

        model, opt, loss_fn, init = self._setup()
        # 4 micro-batches, steps=4: grads accumulate, step fires once at the end.
        with accumulate(opt, steps=4) as acc:
            for _ in range(4):
                x = torch.randn(8, 4)
                y = torch.randn(8, 2)
                acc.backward(loss_fn(model(x), y))
        moved = any(not torch.equal(model.state_dict()[k], init[k]) for k in init)
        assert moved
        # After the block, grads must be cleared (stepped + zeroed).
        assert all(p.grad is None for p in model.parameters())

    def test_partial_flush_on_block_exit(self):
        """If the block exits mid-accumulation, remaining grads must still step."""
        from autotrainer.loop import accumulate

        model, opt, loss_fn, init = self._setup()
        with accumulate(opt, steps=10) as acc:
            # Only 3 backwards, less than steps=10 -> step fires on block exit.
            for _ in range(3):
                x = torch.randn(8, 4)
                y = torch.randn(8, 2)
                acc.backward(loss_fn(model(x), y))
        moved = any(not torch.equal(model.state_dict()[k], init[k]) for k in init)
        assert moved

    def test_rejects_zero_steps(self):
        from autotrainer.loop import accumulate

        opt = torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1)
        with pytest.raises(ValueError, match=">= 1"), accumulate(opt, steps=0):
            pass

    def test_does_not_touch_lr(self):
        """The contract: accumulation scales the step count, not the lr."""
        from autotrainer.loop import accumulate

        model, opt, loss_fn, _ = self._setup()
        original_lr = opt.param_groups[0]["lr"]
        with accumulate(opt, steps=4) as acc:
            for _ in range(4):
                acc.backward(loss_fn(model(torch.randn(8, 4)), torch.randn(8, 2)))
        assert opt.param_groups[0]["lr"] == original_lr
