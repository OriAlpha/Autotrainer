"""Tests for the two knobs added on top of the recipe search: training
length (``epochs``) and image augmentation strength (``aug_strength``).

Covers the augmentation primitive itself (including the no-op paths that let
it sit in a loop that also trains non-vision models), how both knobs enter the
task-aware default space, the per-trial epoch budget, and the plumbing of
``aug_strength`` through to fit()'s phase-2 retrain. All CPU-only.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from autotrainer.augment import MAX_STRENGTH, augment_batch  # noqa: E402
from autotrainer.fitting import fit  # noqa: E402
from autotrainer.tuning import DEFAULT_SPACE, _default_space, tune  # noqa: E402


def _cnn():
    return nn.Sequential(nn.Conv2d(1, 2, 3), nn.Flatten(), nn.Linear(2 * 6 * 6, 3))


def _img_loaders(n=32, batch_size=8):
    """Image-shaped classification loaders (N, 1, 8, 8) -> 3 classes."""
    torch.manual_seed(0)
    x = torch.randn(n, 1, 8, 8)
    y = torch.randint(0, 3, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(
        ds, batch_size=batch_size
    )


def _cls_loaders(n=48, d=4, c=3, batch_size=8):
    torch.manual_seed(0)
    x = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(
        ds, batch_size=batch_size
    )


class TestAugmentBatch:
    def _images(self, n=16):
        torch.manual_seed(0)
        # Strictly positive so any exact zero in the output must come from
        # cutout rather than from the input data.
        return torch.rand(n, 3, 12, 12) + 1.0

    def test_zero_strength_returns_input_unchanged(self):
        x = self._images()
        assert augment_batch(x, 0.0) is x

    def test_negative_strength_is_also_a_noop(self):
        x = self._images()
        assert augment_batch(x, -1.0) is x

    def test_non_image_shapes_pass_through(self):
        # 2D tabular batch: no spatial dims to flip or cut out.
        x = torch.randn(8, 5)
        assert augment_batch(x, MAX_STRENGTH) is x

    def test_integer_tensors_pass_through(self):
        # Token ids are 4D-able in principle but never float; zeroing them
        # would silently mean "token 0", so they must be left alone.
        x = torch.randint(0, 10, (4, 3, 8, 8))
        assert augment_batch(x, MAX_STRENGTH) is x

    def test_non_tensor_passes_through(self):
        batch = {"input_ids": [1, 2, 3]}
        assert augment_batch(batch, MAX_STRENGTH) is batch

    def test_shape_and_dtype_are_preserved(self):
        x = self._images()
        out = augment_batch(x, MAX_STRENGTH)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_input_is_never_mutated(self):
        x = self._images()
        before = x.clone()
        augment_batch(x, MAX_STRENGTH)
        assert torch.equal(x, before)

    def test_cutout_zeroes_some_pixels_at_full_strength(self):
        x = self._images()
        out = augment_batch(x, MAX_STRENGTH)
        # Input was strictly > 1.0, so zeros can only be cutout holes.
        assert (out == 0.0).any()

    def test_cutout_zeroes_whole_channel_columns(self):
        # A hole must cover every channel at a given (h, w) - a per-channel
        # hole would be a different (and wrong) augmentation.
        x = self._images(n=4)
        out = augment_batch(x, MAX_STRENGTH)
        zeroed = out == 0.0
        assert torch.equal(zeroed.all(dim=1), zeroed.any(dim=1))

    def test_strength_is_clamped(self):
        # Well past MAX_STRENGTH must behave exactly like MAX_STRENGTH, not
        # cut a hole bigger than half the image.
        x = self._images()
        torch.manual_seed(1)
        clamped = augment_batch(x, 50.0)
        assert clamped.shape == x.shape
        # A hole of side round(0.5 * 12) = 6 covers at most 36 of 144 px.
        per_image_zeros = (clamped == 0.0).all(dim=1).flatten(1).sum(dim=1)
        assert int(per_image_zeros.max()) <= 36

    def test_small_strength_leaves_most_pixels_intact(self):
        x = self._images()
        out = augment_batch(x, 0.05)
        assert (out != 0.0).float().mean() > 0.9


class TestSpaceEntries:
    def test_epochs_is_always_searched(self):
        assert "epochs" in _default_space(nn.Linear(4, 3), "cross_entropy")
        assert "epochs" in _default_space(nn.Linear(4, 1), "mse")

    def test_epochs_is_bounded_by_the_per_trial_budget(self):
        space = _default_space(nn.Linear(4, 3), "cross_entropy", 7)
        assert space["epochs"] == ("int", 1, 7)

    def test_epochs_bound_is_never_below_one(self):
        space = _default_space(nn.Linear(4, 3), "cross_entropy", 0)
        assert space["epochs"] == ("int", 1, 1)

    def test_aug_strength_only_for_cnns(self):
        assert "aug_strength" in _default_space(_cnn(), "cross_entropy")
        assert "aug_strength" not in _default_space(nn.Linear(4, 3), "cross_entropy")

    def test_aug_strength_range_reaches_max_but_not_past(self):
        assert _default_space(_cnn(), "cross_entropy")["aug_strength"] == (
            "uniform",
            0.0,
            MAX_STRENGTH,
        )

    def test_off_is_reachable_for_augmentation(self):
        # 0.0 must be in range so "no augmentation" is a candidate.
        assert _default_space(_cnn(), "cross_entropy")["aug_strength"][1] == 0.0

    def test_new_keys_are_in_the_maximal_space(self):
        assert {"epochs", "aug_strength"}.issubset(set(DEFAULT_SPACE))

    def test_default_space_stays_a_subset_of_the_maximal_space(self):
        for model in (nn.Linear(4, 3), _cnn()):
            for loss_name in ("cross_entropy", "mse"):
                assert set(_default_space(model, loss_name)).issubset(set(DEFAULT_SPACE))


class TestSearchedEpochs:
    def test_searched_epochs_respect_the_budget(self):
        train, val = _cls_loaders()
        _, best_params, study = tune(
            nn.Linear(4, 3), train, val, trials=4, epochs_per_trial=3, seed=0, verbose=False
        )
        for t in study.trials:
            if "epochs" in t.params:
                assert 1 <= t.params["epochs"] <= 3

    def test_epochs_appears_in_best_params(self):
        train, val = _cls_loaders()
        _, best_params, _ = tune(
            nn.Linear(4, 3), train, val, trials=3, epochs_per_trial=2, seed=0, verbose=False
        )
        assert set(best_params).issubset(set(DEFAULT_SPACE))
        assert "epochs" in best_params

    def test_explicit_space_without_epochs_falls_back_to_the_budget(self):
        # Omitting `epochs` from a custom space must keep the old behavior.
        train, val = _cls_loaders()
        space = {"lr": ("loguniform", 1e-3, 1e-2)}
        _, best_params, _ = tune(
            nn.Linear(4, 3),
            train,
            val,
            trials=2,
            epochs_per_trial=1,
            space=space,
            seed=0,
            verbose=False,
        )
        assert set(best_params) == {"lr"}

    def test_single_epoch_budget_still_runs(self):
        # epochs range collapses to (1, 1); suggest_int must not choke.
        train, val = _cls_loaders()
        model, _, _ = tune(
            nn.Linear(4, 3), train, val, trials=2, epochs_per_trial=1, seed=0, verbose=False
        )
        assert isinstance(model, nn.Module)


class TestAugmentationEndToEnd:
    def test_cnn_tune_runs_with_augmentation_in_the_space(self):
        train, val = _img_loaders()
        _, best_params, _ = tune(
            _cnn(), train, val, trials=3, epochs_per_trial=1, seed=0, verbose=False
        )
        assert "aug_strength" in best_params
        assert 0.0 <= best_params["aug_strength"] <= MAX_STRENGTH

    def test_fit_applies_the_winning_aug_strength(self, monkeypatch):
        seen = []
        import autotrainer.fitting as fitting

        real = fitting.augment_batch

        def spy(x, strength):
            seen.append(strength)
            return real(x, strength)

        monkeypatch.setattr(fitting, "augment_batch", spy)
        train, val = _img_loaders()
        fit(
            _cnn(),
            train,
            val,
            trials=2,
            epochs=1,
            epochs_per_trial=1,
            seed=0,
            verbose=False,
        )
        # Phase 2 ran the augmentation call site, with the searched strength.
        assert seen
        assert all(0.0 <= s <= MAX_STRENGTH for s in seen)

    def test_fit_without_augmentation_passes_zero_strength(self, monkeypatch):
        # Non-CNN -> aug_strength never enters the space -> strength 0.0.
        seen = []
        import autotrainer.fitting as fitting

        real = fitting.augment_batch

        def spy(x, strength):
            seen.append(strength)
            return real(x, strength)

        monkeypatch.setattr(fitting, "augment_batch", spy)
        train, val = _cls_loaders()
        fit(
            nn.Linear(4, 3),
            train,
            val,
            trials=2,
            epochs=1,
            epochs_per_trial=1,
            seed=0,
            verbose=False,
        )
        assert seen
        assert all(s == 0.0 for s in seen)
