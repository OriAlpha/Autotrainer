"""Tests for auto_optim.py: loss inference, optimizer selection, and auto()."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from autotrainer.auto_optim import (  # noqa: E402
    _infer_loss,
    _make_loss,
    _make_optimizer,
    _param_groups,
    auto,
)


class _ToyDataset(Dataset):
    """Configurable toy dataset for loss-inference tests."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TestInferLoss:
    def test_multiclass_int_targets_pick_cross_entropy(self):
        model = nn.Linear(4, 3)  # 3-class output
        yb = torch.tensor([0, 2, 1, 0])
        xb = torch.randn(4, 4)
        loss_fn, name, _ = _infer_loss(model, yb, xb)
        assert name == "cross_entropy"
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_binary_int_targets_single_output_pick_bce(self):
        model = nn.Linear(4, 1)
        yb = torch.tensor([0, 1, 1, 0])
        xb = torch.randn(4, 4)
        loss_fn, name, _ = _infer_loss(model, yb, xb)
        assert name == "bce"
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_single_output_multiclass_not_misrouted_to_bce(self):
        """Regression: a 1-output model with 3+ int classes must NOT pick BCE.

        Previously `out_dim == 1 or n_classes == 2 and out_dim == 1` collapsed
        to `out_dim == 1` due to operator precedence, sending any 1-output
        model to BCE regardless of class count. The fix requires BOTH
        n_classes == 2 AND out_dim == 1.
        """
        model = nn.Linear(4, 1)  # single output
        yb = torch.tensor([0, 1, 2, 1])  # 3 integer classes
        xb = torch.randn(4, 4)
        loss_fn, name, _ = _infer_loss(model, yb, xb)
        assert name == "cross_entropy"
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_clean_float_targets_pick_mse(self):
        model = nn.Linear(3, 1)
        # well-behaved continuous targets, no outliers
        yb = torch.linspace(-1.0, 1.0, 8).unsqueeze(1)
        xb = torch.randn(8, 3)
        loss_fn, name, _ = _infer_loss(model, yb, xb)
        assert name == "mse"
        assert isinstance(loss_fn, nn.MSELoss)

    def test_outlier_float_targets_pick_huber(self):
        model = nn.Linear(3, 1)
        # mostly small values with a few extreme outliers -> Huber
        yb = torch.tensor([0.1, -0.1, 0.2, -0.2, 0.15, 100.0, -100.0, 0.05]).unsqueeze(1)
        xb = torch.randn(8, 3)
        loss_fn, name, _ = _infer_loss(model, yb, xb)
        assert name == "huber"
        assert isinstance(loss_fn, nn.HuberLoss)


class TestMakeLoss:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("cross_entropy", nn.CrossEntropyLoss),
            ("bce", nn.BCEWithLogitsLoss),
            ("mse", nn.MSELoss),
            ("huber", nn.HuberLoss),
        ],
    )
    def test_each_loss_name(self, name, cls):
        assert isinstance(_make_loss(name), cls)


class TestMakeOptimizer:
    def test_cnn_model_picks_sgd(self):
        # A model with a Conv2d should default to SGD (the classic CNN recipe).
        model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(4 * 26 * 26, 2))
        opt, name, _ = _make_optimizer(model, None, lr=0.1, weight_decay=0.0)
        assert name == "sgd"
        assert isinstance(opt, torch.optim.SGD)

    def test_plain_linear_picks_adamw(self):
        model = nn.Linear(5, 2)
        opt, name, _ = _make_optimizer(model, None, lr=0.1, weight_decay=0.0)
        assert name == "adamw"
        assert isinstance(opt, torch.optim.AdamW)

    def test_user_override_respected(self):
        model = nn.Linear(5, 2)
        # Even though it's not a CNN, user asks for sgd explicitly.
        opt, name, reason = _make_optimizer(model, "sgd", lr=0.1, weight_decay=0.0)
        assert name == "sgd"
        assert "override" in reason.lower()


class TestParamGroups:
    def test_bias_and_norm_params_excluded_from_decay(self):
        # bias (Linear) + 1-D BatchNorm weight: both should land in no_decay.
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        groups = _param_groups(model, weight_decay=0.01)
        decay, no_decay = groups[0]["params"], groups[1]["params"]
        # Every param in no_decay must be 1-D (bias/norm) and vice-versa.
        assert all(p.ndim <= 1 for p in no_decay)
        assert all(p.ndim > 1 for p in decay)
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0


class TestAuto:
    def _tiny_loader(self):
        x = torch.randn(16, 3)
        y = torch.randint(0, 2, (16,))  # binary int targets
        return DataLoader(_ToyDataset(x, y), batch_size=4)

    def test_returns_4tuple_without_schedule(self):
        model = nn.Linear(3, 2)
        out = auto(model, self._tiny_loader(), schedule=False)
        assert len(out) == 4
        model_out, loader, opt, loss_fn = out
        assert isinstance(opt, torch.optim.Optimizer)
        assert isinstance(loss_fn, nn.Module)

    def test_returns_5tuple_with_scheduler(self):
        model = nn.Linear(3, 2)
        out = auto(model, self._tiny_loader(), schedule=True, epochs=2)
        assert len(out) == 5
        sched = out[4]
        assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    def test_user_overrides_skip_inference(self, capsys):
        """When loss/optimizer/lr are all overridden, find_lr must not run."""
        model = nn.Linear(3, 2)
        out = auto(
            model,
            self._tiny_loader(),
            loss="mse",
            optimizer="adamw",
            lr=1e-3,
            schedule=False,
        )
        captured = capsys.readouterr().out
        assert "user override" in captured
        # The overridden LR should appear verbatim (not a range-test result).
        assert "1.00e-03" in captured
        assert len(out) == 4
