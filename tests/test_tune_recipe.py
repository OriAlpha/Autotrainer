"""Tests for the expanded tuning recipe.

Covers the pieces added on top of the original lr/wd/optimizer/batch search:
the shared scheduler builder, lr<->batch scaling, the task-aware default
space, label-smoothing plumbing, ASHA as the default pruner, and fit()'s
held-out test estimate. All CPU-only - no cluster or GPU required.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from autotrainer.auto_optim import _make_loss, _make_scheduler, _scale_lr  # noqa: E402
from autotrainer.fitting import fit  # noqa: E402
from autotrainer.tuning import DEFAULT_SPACE, _default_space, tune  # noqa: E402


def _cls_loaders(n=64, d=4, c=3, batch_size=8):
    """Classification loaders (integer targets -> cross-entropy is inferred)."""
    torch.manual_seed(0)
    x = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(
        ds, batch_size=batch_size
    )


def _reg_loaders(n=48, batch_size=8):
    """Three regression loaders (train, val, test) over the same easy task."""
    torch.manual_seed(0)
    x = torch.randn(n, 3)
    y = x.sum(dim=1, keepdim=True)
    ds = TensorDataset(x, y)
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=True),
        DataLoader(ds, batch_size=batch_size),
        DataLoader(ds, batch_size=batch_size),
    )


class TestMakeScheduler:
    def _opt(self, lr=0.1):
        return torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=lr)

    def test_constant_returns_none(self):
        assert _make_scheduler("constant", self._opt(), 100, 0.05) is None

    def test_cosine_with_warmup_is_sequential(self):
        sched = _make_scheduler("cosine", self._opt(), 100, 0.05)
        assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    def test_cosine_zero_warmup_is_bare_cosine(self):
        sched = _make_scheduler("cosine", self._opt(), 100, 0.0)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_onecycle_when_enough_steps(self):
        sched = _make_scheduler("onecycle", self._opt(), 100, 0.05)
        assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)

    def test_onecycle_falls_back_to_cosine_when_too_few_steps(self):
        sched = _make_scheduler("onecycle", self._opt(), 2, 0.05)
        assert not isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)

    def test_onecycle_steps_exactly_total_steps_without_error(self):
        opt = self._opt()
        sched = _make_scheduler("onecycle", opt, 5, 0.05)
        for _ in range(5):  # stepping exactly total_steps must not overrun
            opt.step()
            sched.step()


class TestScaleLr:
    def test_sgd_uses_linear_scaling(self):
        assert _scale_lr(0.1, 64, "sgd") == pytest.approx(0.1 * 64 / 32)

    def test_adamw_uses_sqrt_scaling(self):
        assert _scale_lr(0.1, 128, "adamw") == pytest.approx(0.1 * (128 / 32) ** 0.5)

    def test_reference_batch_is_a_noop(self):
        assert _scale_lr(0.1, 32, "sgd") == 0.1

    def test_mode_none_disables_scaling(self):
        assert _scale_lr(0.1, 128, "sgd", mode="none") == 0.1

    def test_missing_batch_or_optimizer_is_a_noop(self):
        assert _scale_lr(0.1, None, "sgd") == 0.1
        assert _scale_lr(0.1, 128, None) == 0.1

    def test_unknown_optimizer_is_a_noop(self):
        assert _scale_lr(0.1, 128, "lamb") == 0.1


class TestDefaultSpace:
    def _cnn(self):
        return nn.Sequential(nn.Conv2d(1, 2, 3), nn.Flatten(), nn.Linear(2 * 26 * 26, 3))

    def test_cnn_leans_on_onecycle_and_sgd(self):
        space = _default_space(self._cnn(), "cross_entropy")
        assert "onecycle" in space["scheduler"][1]
        assert "sgd" in space["optimizer"][1]

    def test_non_cnn_uses_cosine_or_constant(self):
        space = _default_space(nn.Linear(4, 3), "cross_entropy")
        assert set(space["scheduler"][1]) == {"cosine", "constant"}

    def test_label_smoothing_only_for_cross_entropy(self):
        assert "label_smoothing" in _default_space(nn.Linear(4, 3), "cross_entropy")
        assert "label_smoothing" not in _default_space(nn.Linear(4, 1), "mse")

    def test_keys_are_a_subset_of_the_maximal_space(self):
        for loss_name in ("cross_entropy", "mse"):
            space = _default_space(nn.Linear(4, 3), loss_name)
            assert set(space).issubset(set(DEFAULT_SPACE))


class TestLabelSmoothingLoss:
    def test_cross_entropy_receives_label_smoothing(self):
        assert _make_loss("cross_entropy", label_smoothing=0.1).label_smoothing == pytest.approx(
            0.1
        )

    def test_cross_entropy_default_is_unsmoothed(self):
        assert _make_loss("cross_entropy").label_smoothing == 0.0


class TestTuneExpandedSpace:
    def test_default_pruner_is_asha(self):
        train, val = _cls_loaders()
        _, _, study = tune(nn.Linear(4, 3), train, val, trials=2, epochs_per_trial=1, verbose=False)
        assert isinstance(study.pruner, optuna.pruners.SuccessiveHalvingPruner)

    def test_full_recipe_space_trains_and_returns_valid_params(self):
        train, val = _cls_loaders()
        space = {
            "lr": ("loguniform", 1e-3, 1e-1),
            "optimizer": ("categorical", ["adamw"]),
            "batch_size": ("categorical", [8]),
            "scheduler": ("categorical", ["cosine", "onecycle", "constant"]),
            "warmup_frac": ("uniform", 0.0, 0.1),
            "grad_clip": ("categorical", [0.0, 1.0]),
            "label_smoothing": ("uniform", 0.0, 0.1),
        }
        best_model, best_params, study = tune(
            nn.Linear(4, 3),
            train,
            val,
            trials=4,
            epochs_per_trial=2,
            space=space,
            seed=0,
            verbose=False,
        )
        assert isinstance(best_model, nn.Module)
        assert set(best_params).issubset(set(space))
        assert len(study.trials) == 4

    def test_task_aware_default_space_runs_end_to_end(self):
        # No explicit space -> _default_space kicks in (cross-entropy inferred).
        train, val = _cls_loaders()
        _, best_params, _ = tune(
            nn.Linear(4, 3), train, val, trials=3, epochs_per_trial=1, seed=0, verbose=False
        )
        assert set(best_params).issubset(set(DEFAULT_SPACE))


class TestFitHeldOutTest:
    _SPACE = {"lr": ("loguniform", 1e-3, 1e-1), "optimizer": ("categorical", ["adamw"])}

    def test_test_loss_is_printed_and_stored_on_study(self, capsys):
        train, val, test = _reg_loaders()
        _, _, study = fit(
            nn.Linear(3, 1),
            train,
            val,
            test_loader=test,
            trials=1,
            epochs=2,
            epochs_per_trial=1,
            space=self._SPACE,
            verbose=True,
        )
        assert "test_loss=" in capsys.readouterr().out
        assert "test_loss" in study.user_attrs
        assert math.isfinite(study.user_attrs["test_loss"])

    def test_no_test_loader_leaves_the_attr_unset(self):
        train, val, _ = _reg_loaders()
        _, _, study = fit(
            nn.Linear(3, 1),
            train,
            val,
            trials=1,
            epochs=1,
            epochs_per_trial=1,
            space=self._SPACE,
            verbose=False,
        )
        assert "test_loss" not in study.user_attrs


class TestFitLrScaling:
    def test_auto_and_none_both_train_with_batch_in_space(self):
        # A onecycle schedule + grad clipping in the space also exercises the
        # phase-2 scheduler-choice and (AMP) unscale-then-clip branches.
        space = {
            "lr": ("loguniform", 1e-3, 1e-1),
            "optimizer": ("categorical", ["adamw"]),
            "batch_size": ("categorical", [4]),
            "scheduler": ("categorical", ["onecycle"]),
            "grad_clip": ("categorical", [1.0]),
        }
        for mode in ("auto", "none"):
            train, val, _ = _reg_loaders()
            _, best_params, _ = fit(
                nn.Linear(3, 1),
                train,
                val,
                trials=1,
                epochs=1,
                epochs_per_trial=1,
                space=space,
                lr_scaling=mode,
                verbose=False,
            )
            # best_params always records the un-scaled searched value.
            assert best_params["batch_size"] == 4
