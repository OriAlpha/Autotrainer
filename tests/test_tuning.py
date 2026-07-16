"""Tests for tuning.py: the Optuna hyperparameter search (v0.7 feature)."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from autotrainer.tuning import (  # noqa: E402
    DEFAULT_SPACE,
    _evaluate,
    _rebuild_loader,
    _suggest,
    tune,
)


class TestSharedStorage:
    def test_sequential_tunes_accumulate_in_one_study(self, tmp_path):
        """Two tune() calls on the same journal storage share one study -
        the mechanism fit() uses to run one trial per rank."""
        from autotrainer.fit import _journal_storage

        torch.manual_seed(0)
        x = torch.randn(32, 3)
        y = x.sum(dim=1, keepdim=True)
        train = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
        val = DataLoader(TensorDataset(x, y), batch_size=8)
        model = nn.Linear(3, 1)
        space = {"lr": ("loguniform", 1e-3, 1e-1)}
        path = str(tmp_path / "study.log")

        kwargs = {
            "epochs_per_trial": 1,
            "space": space,
            "loss": "mse",
            "verbose": False,
            "study_name": "shared",
        }
        _, _, study1 = tune(
            model, train, val, trials=2, seed=0, storage=_journal_storage(path), **kwargs
        )
        _, params2, study2 = tune(
            model, train, val, trials=1, seed=1, storage=_journal_storage(path), **kwargs
        )
        assert len(study2.trials) == 3  # 2 + 1 in the SAME study
        assert params2  # best over all three trials
        assert study1.study_name == study2.study_name == "shared"

    def test_zero_trials_returns_empty_params_without_raising(self):
        """A zero-trial share (more ranks than trials) must not crash."""
        torch.manual_seed(0)
        x = torch.randn(16, 3)
        y = x.sum(dim=1, keepdim=True)
        train = DataLoader(TensorDataset(x, y), batch_size=8)
        val = DataLoader(TensorDataset(x, y), batch_size=8)
        _, params, study = tune(nn.Linear(3, 1), train, val, trials=0, loss="mse", verbose=False)
        assert params == {}
        assert len(study.trials) == 0


class TestSuggest:
    def test_all_space_kinds_produce_in_range_values(self):
        space = {
            "lr": ("loguniform", 1e-5, 1e-1),
            "wd": ("uniform", 0.0, 1.0),
            "n": ("int", 1, 10),
            "opt": ("categorical", ["adamw", "sgd"]),
        }
        study = optuna.create_study(direction="minimize")

        def objective(trial):
            p = _suggest(trial, space)
            assert 1e-5 <= p["lr"] <= 1e-1
            assert 0.0 <= p["wd"] <= 1.0
            assert 1 <= p["n"] <= 10
            assert p["opt"] in ("adamw", "sgd")
            return 1.0

        study.optimize(objective, n_trials=1)

    def test_unknown_kind_raises_value_error(self):
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        with pytest.raises(ValueError, match="Unknown space kind"):
            _suggest(trial, {"bad": ("nonsensical", 1, 2)})


class TestRebuildLoader:
    def test_new_batch_size_and_preserved_attrs(self):
        x = torch.randn(20, 3)
        y = torch.randint(0, 2, (20,))
        orig = DataLoader(TensorDataset(x, y), batch_size=10, num_workers=0)
        rebuilt = _rebuild_loader(orig, batch_size=4)
        first_batch = next(iter(rebuilt))
        assert first_batch[0].shape[0] == 4
        assert rebuilt.num_workers == orig.num_workers
        assert rebuilt.drop_last == orig.drop_last


class TestEvaluate:
    def test_returns_finite_float(self):
        model = nn.Linear(3, 2)
        x = torch.randn(8, 3)
        y = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        val = _evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        assert isinstance(val, float)
        assert val == val  # not NaN


class TestTuneEndToEnd:
    def _tiny_setup(self):
        model = nn.Linear(4, 2)
        x = torch.randn(32, 4)
        y = torch.randint(0, 2, (32,))
        train = DataLoader(TensorDataset(x, y), batch_size=8)
        val = DataLoader(TensorDataset(x, y), batch_size=8)
        return model, train, val

    def test_returns_three_tuple_with_valid_params(self):
        model, train, val = self._tiny_setup()
        best_model, best_params, study = tune(
            model, train, val, trials=3, epochs_per_trial=1, seed=0, verbose=False
        )
        assert isinstance(best_model, nn.Module)
        assert isinstance(best_params, dict)
        # Best params should only contain keys from the default search space.
        assert set(best_params.keys()).issubset(set(DEFAULT_SPACE.keys()))
        assert isinstance(study, optuna.Study)
        assert len(study.trials) == 3

    def test_input_model_is_not_mutated(self):
        """The contract: tune() never mutates the user's model object."""
        model, train, val = self._tiny_setup()
        before = copy.deepcopy(model.state_dict())
        tune(model, train, val, trials=2, epochs_per_trial=1, seed=0, verbose=False)
        after = model.state_dict()
        # Every weight tensor must be byte-identical to before the call.
        for key in before:
            assert torch.equal(before[key], after[key]), f"model changed at {key}"

    def test_loss_override_is_used(self, capsys):
        # Regression-shaped model/target so the mse override is shape-compatible
        # (model outputs [B,1], float targets [B,1]).
        model = nn.Linear(4, 1)
        x = torch.randn(32, 4)
        y = torch.randn(32, 1)
        train = DataLoader(TensorDataset(x, y), batch_size=8)
        val = DataLoader(TensorDataset(x, y), batch_size=8)
        best_model, best_params, study = tune(
            model, train, val, trials=2, epochs_per_trial=1, loss="mse", verbose=True
        )
        out = capsys.readouterr().out
        # The override path skips loss inference, so no inference reason should
        # be announced - only the final summary line.
        assert "best val loss" in out
        assert "outliers" not in out  # inference-specific reasoning absent

    def test_restores_optuna_verbosity(self):
        """tune() must not mutate the process-global Optuna log level.

        Optuna's verbosity is module-global; tune() lowers it to WARNING when
        verbose=False. It must restore the prior level on return so callers
        (and other libraries) aren't silently reconfigured.
        """
        import optuna

        model, train, val = self._tiny_setup()
        # Force a known non-default starting level so we can detect restoration.
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        try:
            tune(model, train, val, trials=1, epochs_per_trial=1, verbose=False)
            assert optuna.logging.get_verbosity() == optuna.logging.DEBUG
        finally:
            optuna.logging.set_verbosity(optuna.logging.INFO)
