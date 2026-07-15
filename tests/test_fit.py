"""Tests for fit.py: the tune-then-fully-train orchestrator."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("optuna")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import autotrainer.fit as fit_mod  # noqa: E402
from autotrainer.fit import _unwrap, fit  # noqa: E402


def _loaders(n=64, batch_size=8):
    torch.manual_seed(0)
    x = torch.randn(n, 3)
    y = x.sum(dim=1, keepdim=True)  # easy linear regression
    train = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    val = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    return train, val


# A tiny space keeps the Optuna phase fast and deterministic-ish.
_SMALL_SPACE = {
    "lr": ("loguniform", 1e-3, 1e-1),
    "optimizer": ("categorical", ["adamw"]),
}


class TestFit:
    def test_returns_trained_model_params_and_study(self):
        import optuna

        train, val = _loaders()
        model = nn.Linear(3, 1)
        init = {k: v.clone() for k, v in model.state_dict().items()}

        out_model, best_params, study = fit(
            model,
            train,
            val,
            trials=2,
            epochs=3,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            verbose=False,
        )

        assert isinstance(out_model, nn.Linear)  # plain module, never DDP-wrapped
        assert set(best_params) == {"lr", "optimizer"}
        assert isinstance(study, optuna.study.Study)
        # The input model must be untouched; the returned one must have trained.
        assert all(torch.equal(model.state_dict()[k], init[k]) for k in init)
        assert any(not torch.equal(out_model.state_dict()[k], init[k]) for k in init)

    def test_improves_over_untrained_model(self):
        from autotrainer.tuning import _evaluate

        train, val = _loaders()
        model = nn.Linear(3, 1)
        device = torch.device("cpu")
        before = _evaluate(model, val, nn.MSELoss(), device)

        out_model, _, _ = fit(
            model,
            train,
            val,
            trials=1,
            epochs=5,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            verbose=False,
        )
        after = _evaluate(out_model, val, nn.MSELoss(), device)
        assert after < before

    def test_early_stopping_and_best_weight_restore(self, monkeypatch):
        train, val = _loaders(n=16, batch_size=8)
        model = nn.Linear(3, 1)

        # Phase-2 val losses only get worse after epoch 1 -> with patience=2
        # the loop must run exactly 3 epochs (1 improving + 2 bad) of the 10.
        vals = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        calls = []
        real_evaluate = fit_mod._evaluate

        def fake_evaluate(model, val_loader, loss_fn, device):
            real_evaluate(model, val_loader, loss_fn, device)  # keep it exercised
            v = next(vals)
            calls.append(v)
            return v

        monkeypatch.setattr(fit_mod, "_evaluate", fake_evaluate)
        fit(
            model,
            train,
            val,
            trials=1,
            epochs=10,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            patience=2,
            verbose=False,
        )
        assert calls == [1.0, 2.0, 3.0]

    def test_loss_override_is_respected(self, capsys):
        train, val = _loaders()
        model = nn.Linear(3, 1)
        fit(
            model,
            train,
            val,
            trials=1,
            epochs=1,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            loss="huber",
            verbose=True,
        )
        # With an explicit loss there must be no inference print.
        assert "fit: loss=" not in capsys.readouterr().out

    def test_batch_size_from_space_rebuilds_train_loader(self):
        train, val = _loaders(n=32, batch_size=8)
        model = nn.Linear(3, 1)
        space = dict(_SMALL_SPACE, batch_size=("categorical", [4]))
        _, best_params, _ = fit(
            model,
            train,
            val,
            trials=1,
            epochs=1,
            epochs_per_trial=1,
            space=space,
            verbose=False,
        )
        assert best_params["batch_size"] == 4


class TestCheckpoint:
    def test_saves_every_epoch_and_resumes_without_retuning(self, tmp_path, monkeypatch):
        train, val = _loaders(n=16, batch_size=8)
        model = nn.Linear(3, 1)
        ckpt = str(tmp_path / "fit.ckpt")

        # First run: 2 of 4 epochs, then "preempted" (we just stop at epochs=2).
        _, params_first, study_first = fit(
            model,
            train,
            val,
            trials=1,
            epochs=2,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            checkpoint=ckpt,
            verbose=False,
        )
        assert (tmp_path / "fit.ckpt").exists()
        assert study_first is not None

        # Resume with a larger budget: tuning must be skipped (study None,
        # tune() never called) and training must continue at epoch 3.
        def boom(*a, **k):
            raise AssertionError("tune() must not run on resume")

        monkeypatch.setattr(fit_mod, "tune", boom)
        epochs_run = []
        real_evaluate = fit_mod._evaluate

        def counting_evaluate(model, val_loader, loss_fn, device):
            epochs_run.append(1)
            return real_evaluate(model, val_loader, loss_fn, device)

        monkeypatch.setattr(fit_mod, "_evaluate", counting_evaluate)
        _, params_resumed, study_resumed = fit(
            model,
            train,
            val,
            trials=1,
            epochs=4,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            checkpoint=ckpt,
            verbose=False,
        )
        assert study_resumed is None
        assert params_resumed == params_first
        assert len(epochs_run) == 2  # epochs 3 and 4 only

    def test_resume_of_early_stopped_run_trains_no_further(self, tmp_path, monkeypatch):
        train, val = _loaders(n=16, batch_size=8)
        model = nn.Linear(3, 1)
        ckpt = str(tmp_path / "fit.ckpt")

        # Force val losses that only worsen -> early stop at epoch 3.
        vals = iter([1.0, 2.0, 3.0])
        monkeypatch.setattr(fit_mod, "_evaluate", lambda *a, **k: next(vals))
        fit(
            model,
            train,
            val,
            trials=1,
            epochs=10,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            patience=2,
            checkpoint=ckpt,
            verbose=False,
        )

        # Resuming the stopped run must not train more epochs.
        monkeypatch.setattr(
            fit_mod, "_evaluate", lambda *a, **k: pytest.fail("must not train further")
        )
        out_model, _, _ = fit(
            model,
            train,
            val,
            trials=1,
            epochs=10,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            patience=2,
            checkpoint=ckpt,
            verbose=False,
        )
        assert isinstance(out_model, nn.Linear)


class TestUnwrap:
    def test_plain_module_passes_through(self):
        model = nn.Linear(2, 1)
        assert _unwrap(model) is model

    def test_module_attribute_is_not_mistaken_for_ddp(self):
        # A user model that happens to have a .module attribute must NOT be
        # unwrapped - only real DDP wrappers are.
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.Linear(2, 1)

        w = Wrapper()
        assert _unwrap(w) is w
