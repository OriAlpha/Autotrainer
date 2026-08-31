"""Tests for the data sanity checks.

These warn about the failures that look like a bad recipe and aren't:
un-normalized inputs and class imbalance.
The negative cases matter as much as the positive ones - a check that cries
wolf on a correct setup gets ignored, and then it may as well not exist.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from autotrainer.sanity import report  # noqa: E402


def _loader(ds, batch_size=8):
    return DataLoader(ds, batch_size=batch_size)


def _only(msgs):
    assert len(msgs) == 1, msgs
    return msgs[0]


class TestInputChecks:
    def test_normalized_inputs_are_silent(self):
        assert report(torch.randn(64, 5), torch.randn(64, 1), "mse") == []

    def test_raw_pixels_are_flagged(self):
        x = torch.randint(0, 256, (8, 3, 8, 8)).float()
        assert "raw 0-255 pixels" in _only(report(x, torch.randint(0, 3, (8,)), "cross_entropy"))

    def test_normalized_images_are_not_flagged_as_pixels(self):
        # The check must not fire on data that was already divided by 255.
        x = torch.rand(8, 3, 8, 8)
        assert report(x, torch.randint(0, 3, (8,)), "cross_entropy") == []

    def test_wildly_scaled_inputs_are_flagged(self):
        assert "far from standardized" in _only(
            report(torch.randn(64, 5) * 500, torch.randn(64, 1), "mse")
        )

    def test_non_finite_inputs_are_flagged(self):
        x = torch.tensor([[1.0, float("nan"), 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
        assert "non-finite" in _only(report(x, torch.tensor([0, 1]), "cross_entropy"))

    def test_independent_problems_are_all_reported(self):
        # Input and target checks are separate; one shouldn't mask the other.
        x = torch.tensor([[1.0, float("nan"), 2.0, 3.0]])
        msgs = report(x, torch.tensor([0]), "cross_entropy")
        assert len(msgs) == 2
        assert any("non-finite" in m for m in msgs)
        assert any("same class" in m for m in msgs)

    def test_non_finite_beats_the_other_input_checks(self):
        # NaN is fatal and unconditional; reporting a scale problem alongside
        # it would bury the message that matters.
        x = torch.full((8, 4), float("inf"))
        assert "non-finite" in _only(report(x, torch.randn(8, 1), "mse"))

    def test_constant_inputs_are_flagged(self):
        assert "essentially constant" in _only(
            report(torch.zeros(16, 4), torch.randn(16, 1), "mse")
        )

    def test_non_tensor_inputs_are_skipped(self):
        assert report({"input_ids": [1, 2, 3]}, torch.randn(4, 1), "mse") == []

    def test_integer_inputs_are_skipped(self):
        # Token ids have no meaningful mean/std to complain about.
        assert (
            report(torch.randint(0, 5000, (8, 32)), torch.randint(0, 3, (8,)), "cross_entropy")
            == []
        )


class TestTargetChecks:
    def test_balanced_classes_are_silent(self):
        y = torch.tensor([0] * 100 + [1] * 100)
        assert report(torch.randn(200, 4), y, "cross_entropy") == []

    def test_mild_imbalance_is_silent(self):
        # 3:1 is ordinary; warning about it would be noise.
        y = torch.tensor([0] * 150 + [1] * 50)
        assert report(torch.randn(200, 4), y, "cross_entropy") == []

    def test_severe_imbalance_is_flagged_with_the_baseline(self):
        y = torch.tensor([0] * 190 + [1] * 10)
        msg = _only(report(torch.randn(200, 4), y, "cross_entropy"))
        assert "19:1" in msg
        assert "95%" in msg  # the accuracy a majority-only predictor gets
        assert "class weights" in msg  # and the way out

    def test_message_says_how_much_data_it_looked_at(self):
        # The checks run on a sample; the message must not imply otherwise.
        y = torch.tensor([0] * 190 + [1] * 10)
        assert "200 targets sampled" in _only(report(torch.randn(200, 4), y, "cross_entropy"))

    def test_single_class_is_flagged(self):
        y = torch.zeros(20, dtype=torch.long)
        assert "same class" in _only(report(torch.randn(20, 4), y, "cross_entropy"))

    def test_constant_regression_targets_are_flagged(self):
        y = torch.full((20, 1), 3.0)
        assert "same value" in _only(report(torch.randn(20, 4), y, "mse"))

    def test_varied_regression_targets_are_silent(self):
        assert report(torch.randn(20, 4), torch.randn(20, 1), "mse") == []

    def test_non_finite_targets_are_flagged(self):
        y = torch.tensor([1.0, float("nan"), 3.0])
        assert "non-finite" in _only(report(torch.randn(3, 4), y, "mse"))

    def test_multilabel_targets_are_left_alone(self):
        # (N, C) 0/1 targets are a different question than class balance.
        y = torch.randint(0, 2, (20, 6)).float()
        assert report(torch.randn(20, 4), y, "bce") == []

    def test_class_checks_do_not_run_on_regression(self):
        # Continuous targets happen to be non-negative here; they must not be
        # read as wildly imbalanced class labels.
        y = torch.rand(200, 1) * 0.001
        assert report(torch.randn(200, 4), y, "mse") == []

    def test_float_class_labels_are_understood(self):
        y = torch.tensor([0.0] * 190 + [1.0] * 10)
        assert "imbalance" in _only(report(torch.randn(200, 4), y, "bce"))


class TestWiring:
    """The checks have to actually reach the user through auto()."""

    def test_auto_warns(self, capsys):
        from autotrainer.auto_optim import auto

        torch.manual_seed(0)
        ds = TensorDataset(torch.randn(32, 4) * 500, torch.randint(0, 2, (32,)))
        auto(nn.Linear(4, 2), _loader(ds), lr=1e-3, schedule=False)
        assert "far from standardized" in capsys.readouterr().out

    def test_auto_can_turn_them_off(self, capsys):
        from autotrainer.auto_optim import auto

        torch.manual_seed(0)
        ds = TensorDataset(torch.randn(32, 4) * 500, torch.randint(0, 2, (32,)))
        auto(nn.Linear(4, 2), _loader(ds), lr=1e-3, schedule=False, sanity=False)
        assert "data check" not in capsys.readouterr().out
