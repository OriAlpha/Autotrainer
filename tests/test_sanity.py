"""Tests for the data sanity checks.

These warn about the failures that look like a bad recipe and aren't:
un-normalized inputs, class imbalance, and a train/val split that overlaps.
The negative cases matter as much as the positive ones - a check that cries
wolf on a correct setup gets ignored, and then it may as well not exist.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split  # noqa: E402

from autotrainer.sanity import overlap, report  # noqa: E402


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
        assert "f1" in msg  # and the way out

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


class TestOverlap:
    def _ds(self, n=200, d=6):
        torch.manual_seed(0)
        return TensorDataset(torch.randn(n, d), torch.randint(0, 3, (n,)))

    def test_clean_random_split_is_silent(self):
        # The check has to stay quiet on the correct way to do this, or it
        # will be tuned out.
        ds = self._ds()
        tr, va = random_split(ds, [160, 40], generator=torch.Generator().manual_seed(0))
        assert overlap(_loader(tr), _loader(va)) == []

    def test_disjoint_datasets_are_silent(self):
        torch.manual_seed(1)
        tr = TensorDataset(torch.randn(150, 6), torch.randint(0, 3, (150,)))
        va = TensorDataset(torch.randn(50, 6), torch.randint(0, 3, (50,)))
        assert overlap(_loader(tr), _loader(va)) == []

    def test_same_dataset_object_is_flagged(self):
        ds = self._ds()
        assert "SAME dataset object" in _only(overlap(_loader(ds), _loader(ds)))

    def test_overlapping_subsets_report_the_exact_count(self):
        ds = self._ds()
        tr = Subset(ds, list(range(0, 180)))
        va = Subset(ds, list(range(150, 200)))
        msg = _only(overlap(_loader(tr), _loader(va)))
        assert "share 30 of the validation set's 50 samples" in msg

    def test_copied_rows_are_caught_by_value(self):
        # Distinct dataset objects, but the val rows came out of train.
        ds = self._ds()
        x, y = ds.tensors
        tr = TensorDataset(x[:150], y[:150])
        va = TensorDataset(x[100:150].clone(), y[100:150].clone())
        assert "identical to rows" in _only(overlap(_loader(tr), _loader(va)))

    def test_low_dimensional_data_is_not_value_checked(self):
        # With 2 features, identical rows arise by chance; a value-based
        # verdict there would be a false alarm.
        torch.manual_seed(0)
        tr = TensorDataset(torch.randint(0, 3, (150, 2)).float(), torch.randint(0, 2, (150,)))
        va = TensorDataset(torch.randint(0, 3, (50, 2)).float(), torch.randint(0, 2, (50,)))
        assert overlap(_loader(tr), _loader(va)) == []

    def test_a_single_coincidence_is_not_reported(self):
        torch.manual_seed(0)
        x = torch.randn(150, 6)
        tr = TensorDataset(x, torch.randint(0, 3, (150,)))
        va_x = torch.randn(50, 6)
        va_x[0] = x[0]  # exactly one shared row
        va = TensorDataset(va_x, torch.randint(0, 3, (50,)))
        assert overlap(_loader(tr), _loader(va)) == []

    def test_loaders_without_datasets_are_skipped(self):
        assert overlap(object(), object()) == []

    def test_streaming_datasets_are_never_consumed(self):
        # A check that eats the data it is checking is worse than no check.
        from torch.utils.data import IterableDataset

        class Stream(IterableDataset):
            def __init__(self, n):
                self.n = n
                self.reads = 0

            def __iter__(self):
                self.reads += 1
                for _ in range(self.n):
                    yield torch.randn(6), torch.randint(0, 3, ()).long()

        tr, va = Stream(64), Stream(16)
        assert overlap(_loader(tr), _loader(va)) == []
        assert (tr.reads, va.reads) == (0, 0)


class TestWiring:
    """The checks have to actually reach the user through auto()/tune()."""

    def test_tune_warns_before_searching(self, capsys):
        pytest.importorskip("optuna")
        from autotrainer.tuning import tune

        torch.manual_seed(0)
        # Same dataset for train and val: the overlap check must fire.
        ds = TensorDataset(torch.randn(48, 4), torch.tensor([0] * 44 + [1] * 4))
        train, val = _loader(ds), _loader(ds)
        tune(
            nn.Linear(4, 2),
            train,
            val,
            trials=1,
            epochs_per_trial=1,
            space={"lr": ("loguniform", 1e-3, 1e-2)},
            seed=0,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert "SAME dataset object" in out
        assert "class imbalance" in out

    def test_tune_can_turn_them_off(self, capsys):
        pytest.importorskip("optuna")
        from autotrainer.tuning import tune

        torch.manual_seed(0)
        ds = TensorDataset(torch.randn(48, 4), torch.tensor([0] * 44 + [1] * 4))
        tune(
            nn.Linear(4, 2),
            _loader(ds),
            _loader(ds),
            trials=1,
            epochs_per_trial=1,
            space={"lr": ("loguniform", 1e-3, 1e-2)},
            seed=0,
            verbose=True,
            sanity=False,
        )
        assert "data check" not in capsys.readouterr().out

    def test_quiet_tune_stays_quiet(self, capsys):
        pytest.importorskip("optuna")
        from autotrainer.tuning import tune

        torch.manual_seed(0)
        ds = TensorDataset(torch.randn(48, 4), torch.tensor([0] * 44 + [1] * 4))
        tune(
            nn.Linear(4, 2),
            _loader(ds),
            _loader(ds),
            trials=1,
            epochs_per_trial=1,
            space={"lr": ("loguniform", 1e-3, 1e-2)},
            seed=0,
            verbose=False,
        )
        assert capsys.readouterr().out == ""

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

    def test_clean_data_adds_no_noise_to_tune(self, capsys):
        pytest.importorskip("optuna")
        from autotrainer.tuning import tune

        torch.manual_seed(0)
        x, y = torch.randn(200, 6), torch.randint(0, 3, (200,))
        tr, va = random_split(
            TensorDataset(x, y), [160, 40], generator=torch.Generator().manual_seed(0)
        )
        tune(
            nn.Linear(6, 3),
            _loader(tr),
            _loader(va),
            trials=1,
            epochs_per_trial=1,
            space={"lr": ("loguniform", 1e-3, 1e-2)},
            seed=0,
            verbose=True,
        )
        assert "data check" not in capsys.readouterr().out
