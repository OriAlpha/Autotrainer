"""Tests for metric-aware selection and preemption handling.

Two features that both change *when a run stops and what it hands back*:

  * ``metric=`` - tune()/fit() select, prune, early-stop and pick the best
    epoch on the number you name, not always on the loss;
  * preemption - a watched SIGUSR1/SIGTERM stops phase 2 at an epoch
    boundary with the checkpoint already written, and the phase-1 search is
    journaled so it resumes instead of restarting.

The metric maths is checked against values worked out by hand (including the
tie case, which is where a naive AUC silently inflates). All CPU-only.
"""

from __future__ import annotations

import signal

import pytest

torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import autotrainer.fitting as fit_mod  # noqa: E402
from autotrainer import preempt  # noqa: E402
from autotrainer._fit_search import _journal_storage  # noqa: E402
from autotrainer.fitting import fit  # noqa: E402
from autotrainer.metrics import (  # noqa: E402
    MAXIMIZE,
    MINIMIZE,
    is_better,
    name_of,
    resolve,
    score,
    worst,
)
from autotrainer.tuning import _evaluate, tune  # noqa: E402

CPU = torch.device("cpu")

# A tiny space keeps the Optuna phase fast.
_SMALL_SPACE = {
    "lr": ("loguniform", 1e-3, 1e-1),
    "optimizer": ("categorical", ["adamw"]),
}


def _scored(x, y, batch_size=8):
    """A loader whose inputs ARE the model outputs, paired with nn.Identity().

    Lets a metric be checked against exact hand-computed numbers instead of
    whatever a real model happens to predict.
    """
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _cls_loaders(n=48, d=4, c=3, batch_size=8):
    torch.manual_seed(0)
    x = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(
        ds, batch_size=batch_size
    )


def _reg_loaders(n=64, batch_size=8):
    torch.manual_seed(0)
    x = torch.randn(n, 3)
    y = x.sum(dim=1, keepdim=True)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True), DataLoader(
        TensorDataset(x, y), batch_size=batch_size
    )


class TestResolve:
    def test_none_and_loss_both_mean_loss(self):
        assert resolve(None) == ("loss", MINIMIZE)
        assert resolve("loss") == ("loss", MINIMIZE)

    @pytest.mark.parametrize("name", ["accuracy", "f1", "auc", "r2"])
    def test_score_metrics_are_maximized(self, name):
        assert resolve(name) == (name, MAXIMIZE)

    def test_unknown_metric_raises_rather_than_falling_back(self):
        # A typo must not quietly leave the search optimizing the loss.
        with pytest.raises(ValueError, match="Unknown metric"):
            resolve("accuarcy")

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            resolve("accuracy", "biggest")

    def test_callable_defaults_to_maximize(self):
        def my_scorer(model, loader):
            return 1.0

        metric, direction = resolve(my_scorer)
        assert metric is my_scorer
        assert direction == MAXIMIZE

    def test_direction_can_be_overridden(self):
        assert resolve(lambda m, dl: 0.0, MINIMIZE)[1] == MINIMIZE
        assert resolve("loss", MAXIMIZE)[1] == MAXIMIZE

    def test_name_of(self):
        assert name_of("f1") == "f1"

        def custom(model, loader):
            return 0.0

        assert name_of(custom) == "custom"


class TestDirectionHelpers:
    def test_worst_is_beaten_by_anything(self):
        assert is_better(-1e9, worst(MAXIMIZE), MAXIMIZE)
        assert is_better(1e9, worst(MINIMIZE), MINIMIZE)

    def test_min_delta_is_a_magnitude_in_both_directions(self):
        # The same setting must mean the same size of movement either way.
        assert not is_better(0.55, 0.5, MAXIMIZE, min_delta=0.1)
        assert is_better(0.65, 0.5, MAXIMIZE, min_delta=0.1)
        assert not is_better(0.45, 0.5, MINIMIZE, min_delta=0.1)
        assert is_better(0.35, 0.5, MINIMIZE, min_delta=0.1)


class TestAccuracy:
    def test_multiclass_argmax(self):
        x = torch.tensor([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0]])  # -> 0, 1, 0
        y = torch.tensor([0, 1, 1])
        assert score(nn.Identity(), _scored(x, y), "accuracy", CPU) == pytest.approx(2 / 3)

    def test_single_logit_is_thresholded_at_zero(self):
        # BCE-style head: one logit per sample, positive means class 1.
        x = torch.tensor([[1.5], [-0.5], [0.2], [-2.0]])  # -> 1, 0, 1, 0
        y = torch.tensor([1, 0, 0, 0])
        assert score(nn.Identity(), _scored(x, y), "accuracy", CPU) == pytest.approx(0.75)

    def test_float_targets_are_accepted(self):
        x = torch.tensor([[1.5], [-0.5]])
        y = torch.tensor([[1.0], [0.0]])
        assert score(nn.Identity(), _scored(x, y), "accuracy", CPU) == pytest.approx(1.0)

    def test_batching_does_not_change_the_answer(self):
        x = torch.tensor([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0], [0.0, 1.0]])
        y = torch.tensor([0, 1, 0, 0])
        whole = score(nn.Identity(), _scored(x, y, batch_size=4), "accuracy", CPU)
        split = score(nn.Identity(), _scored(x, y, batch_size=1), "accuracy", CPU)
        assert whole == pytest.approx(split) == pytest.approx(0.75)


class TestF1:
    def test_macro_f1_matches_hand_computation(self):
        # preds 0,1,1,1,0 vs true 0,0,1,1,1
        #   class 0: tp=1 fp=1 fn=1 -> 0.5      class 1: tp=2 fp=1 fn=1 -> 2/3
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y = torch.tensor([0, 0, 1, 1, 1])
        expected = (0.5 + 2 / 3) / 2
        assert score(nn.Identity(), _scored(x, y), "f1", CPU) == pytest.approx(expected)

    def test_absent_classes_do_not_dilute_the_average(self):
        # A 3-wide head but only classes 0/1 in the targets: a perfect
        # prediction is 1.0, not 2/3.
        x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y = torch.tensor([0, 1])
        assert score(nn.Identity(), _scored(x, y), "f1", CPU) == pytest.approx(1.0)

    def test_majority_predictor_scores_poorly_on_imbalance(self):
        # The whole point of macro F1: 9:1 imbalance, always predict the
        # majority -> accuracy 0.9 but F1 far lower.
        x = torch.tensor([[1.0, 0.0]] * 10)
        y = torch.tensor([0] * 9 + [1])
        loader = _scored(x, y)
        assert score(nn.Identity(), loader, "accuracy", CPU) == pytest.approx(0.9)
        assert score(nn.Identity(), loader, "f1", CPU) < 0.5


class TestAuc:
    def test_binary_auc_matches_hand_computation(self):
        x = torch.tensor([[0.1], [0.4], [0.35], [0.8]])
        y = torch.tensor([0, 0, 1, 1])
        assert score(nn.Identity(), _scored(x, y), "auc", CPU) == pytest.approx(0.75)

    def test_ties_get_average_ranks(self):
        # Two positives and two negatives sharing two score values is a coin
        # flip (0.5). Ignoring ties would report 0.75 or 0.25 instead.
        x = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        y = torch.tensor([1, 0, 1, 0])
        assert score(nn.Identity(), _scored(x, y), "auc", CPU) == pytest.approx(0.5)

    def test_perfect_ranking_is_one(self):
        x = torch.tensor([[0.0], [0.1], [0.9], [1.0]])
        y = torch.tensor([0, 0, 1, 1])
        assert score(nn.Identity(), _scored(x, y), "auc", CPU) == pytest.approx(1.0)

    def test_single_class_validation_set_raises_clearly(self):
        x = torch.tensor([[0.1], [0.4]])
        y = torch.tensor([1, 1])
        with pytest.raises(ValueError, match="auc is undefined"):
            score(nn.Identity(), _scored(x, y), "auc", CPU)

    def test_multiclass_one_vs_rest(self):
        x = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
        y = torch.tensor([0, 1, 2])
        assert score(nn.Identity(), _scored(x, y), "auc", CPU) == pytest.approx(1.0)

    def test_one_vs_rest_ranks_on_probabilities_not_raw_logits(self):
        # Softmax's denominator varies per sample, so it is NOT a monotone
        # transform of a single logit column: ranking class c by its raw logit
        # gives a different (and wrong) AUC. This case separates them -
        # 0.5833... is what the logit version would report.
        x = torch.tensor([[2.0, 0.0, 0.0], [1.0, -5.0, -5.0], [0.5, 0.0, 0.0], [3.0, 3.0, 3.0]])
        y = torch.tensor([1, 0, 2, 0])
        assert score(nn.Identity(), _scored(x, y), "auc", CPU) == pytest.approx(0.5)

    def test_non_probability_outputs_beyond_2d_raise(self):
        x = torch.randn(4, 2, 3, 3)
        y = torch.tensor([0, 1, 0, 1])
        with pytest.raises(ValueError, match="expected"):
            score(nn.Identity(), _scored(x, y), "auc", CPU)


class TestR2:
    def test_perfect_fit_is_one(self):
        y = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        assert score(nn.Identity(), _scored(y, y), "r2", CPU) == pytest.approx(1.0)

    def test_mean_predictor_is_zero(self):
        y = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        x = torch.full_like(y, 2.5)
        assert score(nn.Identity(), _scored(x, y), "r2", CPU) == pytest.approx(0.0)

    def test_worse_than_the_mean_is_negative(self):
        y = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        x = torch.tensor([[4.0], [3.0], [2.0], [1.0]])
        assert score(nn.Identity(), _scored(x, y), "r2", CPU) < 0.0

    def test_constant_targets_do_not_divide_by_zero(self):
        y = torch.full((4, 1), 2.0)
        assert score(nn.Identity(), _scored(y, y), "r2", CPU) == pytest.approx(1.0)
        assert score(nn.Identity(), _scored(torch.zeros(4, 1), y), "r2", CPU) == pytest.approx(0.0)


class TestMetricErrors:
    """The messages a user meets when a metric is pointed at the wrong task.

    These have to name the problem and the way out, because reaching them
    means the run is about to stop.
    """

    def _empty(self, y_dtype=torch.long):
        return DataLoader(
            TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=y_dtype)), batch_size=8
        )

    def test_targets_that_do_not_line_up_with_predictions(self):
        x = torch.randn(4, 1)  # -> one prediction per sample
        y = torch.randint(0, 2, (4, 2))  # but two targets per sample
        with pytest.raises(ValueError, match="don't line up"):
            score(nn.Identity(), _scored(x, y), "accuracy", CPU)

    def test_negative_class_targets(self):
        x = torch.randn(4, 1)
        y = torch.tensor([-1, 0, 1, 0])
        with pytest.raises(ValueError, match="non-negative integer class targets"):
            score(nn.Identity(), _scored(x, y), "accuracy", CPU)

    def test_binary_auc_rejects_non_binary_labels(self):
        x = torch.randn(4, 1)
        y = torch.tensor([0, 2, 1, 0])
        with pytest.raises(ValueError, match="targets must be 0/1"):
            score(nn.Identity(), _scored(x, y), "auc", CPU)

    def test_r2_rejects_mismatched_sizes(self):
        x = torch.randn(4, 2)
        y = torch.randn(4, 1)
        with pytest.raises(ValueError, match="r2:"):
            score(nn.Identity(), _scored(x, y), "r2", CPU)

    @pytest.mark.parametrize("metric", ["accuracy", "f1", "auc", "r2"])
    def test_empty_validation_set(self, metric):
        with pytest.raises(ValueError, match="no samples"):
            score(nn.Identity(), self._empty(), metric, CPU)


class TestClassCountsAcrossBatches:
    def test_later_batches_with_fewer_classes_still_accumulate(self):
        # Per-class counters are sized by the classes seen so far, so a batch
        # covering fewer classes than an earlier one must widen back out
        # rather than truncate the running totals.
        x = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # -> 2, then 0
        y = torch.tensor([2, 0])
        assert score(nn.Identity(), _scored(x, y, batch_size=1), "f1", CPU) == pytest.approx(1.0)


def _boom(*a, **k):
    raise AssertionError("metrics must not call the criterion")


class TestEvaluateDispatch:
    def test_loss_path_is_unchanged(self):
        _, val = _cls_loaders()
        model = nn.Linear(4, 3)
        plain = _evaluate(model, val, nn.CrossEntropyLoss(), CPU)
        explicit = _evaluate(model, val, nn.CrossEntropyLoss(), CPU, "loss")
        assert plain == pytest.approx(explicit)

    def test_metric_path_ignores_the_criterion(self):
        _, val = _cls_loaders()
        acc = _evaluate(nn.Linear(4, 3), val, _boom, CPU, "accuracy")
        assert 0.0 <= acc <= 1.0

    def test_targetless_batches_give_a_clear_error(self):
        loader = DataLoader(TensorDataset(torch.randn(8, 3)), batch_size=4)
        with pytest.raises(ValueError, match="needs targets"):
            _evaluate(nn.Identity(), loader, nn.MSELoss(), CPU, "accuracy")


class TestTuneWithMetric:
    def test_study_direction_follows_the_metric(self):
        train, val = _cls_loaders()
        _, _, study = tune(
            nn.Linear(4, 3),
            train,
            val,
            trials=2,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            metric="accuracy",
            seed=0,
            verbose=False,
        )
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE
        assert 0.0 <= study.best_value <= 1.0

    def test_loss_still_minimizes_by_default(self):
        train, val = _cls_loaders()
        _, _, study = tune(
            nn.Linear(4, 3),
            train,
            val,
            trials=2,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            seed=0,
            verbose=False,
        )
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_best_trial_is_the_highest_scoring_one(self):
        train, val = _cls_loaders()
        _, _, study = tune(
            nn.Linear(4, 3),
            train,
            val,
            trials=4,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            metric="accuracy",
            seed=0,
            verbose=False,
        )
        done = [t.value for t in study.trials if t.value is not None]
        assert study.best_value == pytest.approx(max(done))

    def test_custom_callable_is_used(self):
        seen = []

        def constant_scorer(model, loader):
            seen.append(1)
            return 0.5

        train, val = _cls_loaders()
        _, _, study = tune(
            nn.Linear(4, 3),
            train,
            val,
            trials=2,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            metric=constant_scorer,
            seed=0,
            verbose=False,
        )
        assert seen
        assert study.best_value == pytest.approx(0.5)
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_unknown_metric_is_rejected_before_any_training(self):
        train, val = _cls_loaders()
        with pytest.raises(ValueError, match="Unknown metric"):
            tune(nn.Linear(4, 3), train, val, trials=1, metric="acc", verbose=False)

    def test_summary_line_names_the_metric(self, capsys):
        train, val = _cls_loaders()
        tune(
            nn.Linear(4, 3),
            train,
            val,
            trials=1,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            metric="accuracy",
            seed=0,
            verbose=True,
        )
        assert "best val accuracy" in capsys.readouterr().out


class TestFitWithMetric:
    def test_early_stopping_flips_direction(self, monkeypatch):
        # Accuracies that only get WORSE after epoch 1. Under a maximized
        # metric that must stop after 1 improving + 2 bad epochs - the same
        # shape as the loss test, in the opposite direction.
        train, val = _cls_loaders()
        vals = iter([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        calls = []

        def fake_evaluate(model, val_loader, loss_fn, device, metric=None):
            v = next(vals)
            calls.append(v)
            return v

        monkeypatch.setattr(fit_mod, "_evaluate", fake_evaluate)
        fit(
            nn.Linear(4, 3),
            train,
            val,
            trials=1,
            epochs=10,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            patience=2,
            metric="accuracy",
            verbose=False,
        )
        assert calls == [0.9, 0.8, 0.7]

    def test_best_epoch_is_the_peak_not_the_last(self, monkeypatch, capsys):
        # The peak is at epoch 2. A loss-shaped comparison would treat the
        # falling scores as improvements and keep the LAST epoch instead.
        train, val = _cls_loaders()
        vals = iter([0.5, 0.9, 0.4, 0.3])

        def fake_evaluate(model, val_loader, loss_fn, device, metric=None):
            return next(vals)

        monkeypatch.setattr(fit_mod, "_evaluate", fake_evaluate)
        fit(
            nn.Linear(4, 3),
            train,
            val,
            trials=1,
            epochs=4,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            patience=10,
            metric="accuracy",
            verbose=True,
        )
        out = capsys.readouterr().out
        assert "best val_accuracy=0.9000" in out
        # Only the first two epochs count as improvements.
        assert out.count("val_accuracy=0.5000 *") == 1
        assert out.count("val_accuracy=0.9000 *") == 1
        assert "val_accuracy=0.4000 *" not in out

    def test_epoch_lines_and_summary_name_the_metric(self, monkeypatch, capsys):
        train, val = _cls_loaders()

        def fake_evaluate(model, val_loader, loss_fn, device, metric=None):
            return 0.5

        monkeypatch.setattr(fit_mod, "_evaluate", fake_evaluate)
        fit(
            nn.Linear(4, 3),
            train,
            val,
            trials=1,
            epochs=1,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            metric="accuracy",
            verbose=True,
        )
        out = capsys.readouterr().out
        assert "val_accuracy=" in out
        assert "val_loss=" not in out

    def test_test_score_is_stored_under_both_names_for_loss(self):
        train, val = _reg_loaders(n=16, batch_size=8)
        _, _, study = fit(
            nn.Linear(3, 1),
            train,
            val,
            test_loader=val,
            trials=1,
            epochs=1,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            verbose=False,
        )
        # "test_loss" is the pre-metric name and must keep working.
        assert study.user_attrs["test_score"] == pytest.approx(study.user_attrs["test_loss"])

    def test_test_score_uses_the_metric(self):
        train, val = _cls_loaders(n=32)
        _, _, study = fit(
            nn.Linear(4, 3),
            train,
            val,
            test_loader=val,
            trials=1,
            epochs=1,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            metric="accuracy",
            verbose=False,
        )
        assert 0.0 <= study.user_attrs["test_score"] <= 1.0
        assert "test_loss" not in study.user_attrs  # not a loss, don't call it one

    def test_metric_reaches_the_search(self):
        # end-to-end: fit() with a metric produces a study that maximized it.
        train, val = _cls_loaders(n=32)
        _, _, study = fit(
            nn.Linear(4, 3),
            train,
            val,
            trials=2,
            epochs=1,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            metric="accuracy",
            verbose=False,
        )
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE


class TestCheckpointMetricGuard:
    def test_resuming_with_a_different_metric_is_refused(self, tmp_path):
        train, val = _reg_loaders(n=16, batch_size=8)
        ckpt = str(tmp_path / "run.pt")
        fit(
            nn.Linear(3, 1),
            train,
            val,
            trials=1,
            epochs=1,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            checkpoint=ckpt,
            verbose=False,
        )
        with pytest.raises(ValueError, match="not comparable"):
            fit(
                nn.Linear(3, 1),
                train,
                val,
                trials=1,
                epochs=2,
                epochs_per_trial=1,
                space=_SMALL_SPACE,
                checkpoint=ckpt,
                metric="accuracy",
                verbose=False,
            )

    def test_resuming_with_the_same_metric_works(self, tmp_path, monkeypatch):
        train, val = _cls_loaders()
        ckpt = str(tmp_path / "run.pt")

        def fake_evaluate(model, val_loader, loss_fn, device, metric=None):
            return 0.5

        monkeypatch.setattr(fit_mod, "_evaluate", fake_evaluate)
        for epochs in (1, 2):
            fit(
                nn.Linear(4, 3),
                train,
                val,
                trials=1,
                epochs=epochs,
                epochs_per_trial=1,
                space=_SMALL_SPACE,
                checkpoint=ckpt,
                metric="accuracy",
                verbose=False,
            )
        stored = torch.load(ckpt, map_location="cpu", weights_only=True)
        assert stored["metric"] == "accuracy"
        assert stored["direction"] == "maximize"


class TestPreemptModule:
    @pytest.fixture(autouse=True)
    def _clean(self):
        preempt.reset()
        yield
        preempt.reset()

    def test_flag_starts_clear(self):
        assert preempt.preempted() is False

    def test_handler_sets_the_flag_without_raising(self):
        preempt.watch()
        preempt._handler(signal.SIGTERM, None)  # what the OS would deliver
        assert preempt.preempted() is True

    def test_watch_installs_and_restore_puts_back(self):
        original = signal.getsignal(signal.SIGTERM)
        installed = preempt.watch()
        assert "SIGTERM" in installed
        assert signal.getsignal(signal.SIGTERM) is preempt._handler
        preempt.restore()
        assert signal.getsignal(signal.SIGTERM) is original

    def test_watch_is_idempotent(self):
        first = preempt.watch()
        second = preempt.watch()
        assert "SIGTERM" in first
        assert second == []  # nothing re-installed, original handler not lost

    def test_unknown_signal_names_are_skipped(self):
        # SIGUSR1 does not exist on Windows; asking for a missing signal must
        # not blow up a training run.
        assert preempt.watch(("SIGNOPE",)) == []

    def test_reset_clears_the_flag(self):
        preempt._handler(signal.SIGTERM, None)
        preempt.reset()
        assert preempt.preempted() is False


class TestFitPreemption:
    @pytest.fixture(autouse=True)
    def _clean(self):
        preempt.reset()
        yield
        preempt.reset()

    def test_run_stops_at_the_epoch_boundary_with_a_checkpoint(self, tmp_path, monkeypatch):
        import os

        train, val = _reg_loaders(n=16, batch_size=8)
        ckpt = str(tmp_path / "run.pt")
        epochs_run = []
        real_evaluate = fit_mod._evaluate

        def evaluate_then_preempt(model, val_loader, loss_fn, device, metric=None):
            epochs_run.append(1)
            preempt._handler(signal.SIGTERM, None)
            return real_evaluate(model, val_loader, loss_fn, device, metric)

        monkeypatch.setattr(fit_mod, "_evaluate", evaluate_then_preempt)
        fit(
            nn.Linear(3, 1),
            train,
            val,
            trials=1,
            epochs=10,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            checkpoint=ckpt,
            verbose=False,
        )
        assert len(epochs_run) == 1  # stopped at the first boundary, not after 10
        assert os.path.exists(ckpt)  # and the epoch it stopped on is durable

    def test_the_checkpoint_written_on_preemption_resumes(self, tmp_path, monkeypatch):
        train, val = _reg_loaders(n=16, batch_size=8)
        ckpt = str(tmp_path / "run.pt")
        real_evaluate = fit_mod._evaluate

        def evaluate_then_preempt(model, val_loader, loss_fn, device, metric=None):
            preempt._handler(signal.SIGTERM, None)
            return real_evaluate(model, val_loader, loss_fn, device, metric)

        monkeypatch.setattr(fit_mod, "_evaluate", evaluate_then_preempt)
        fit(
            nn.Linear(3, 1),
            train,
            val,
            trials=1,
            epochs=4,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            checkpoint=ckpt,
            verbose=False,
        )
        # The requeued run picks up where it stopped rather than at epoch 0.
        preempt.reset()
        epochs_run = []

        def counting(model, val_loader, loss_fn, device, metric=None):
            epochs_run.append(1)
            return real_evaluate(model, val_loader, loss_fn, device, metric)

        monkeypatch.setattr(fit_mod, "_evaluate", counting)
        fit(
            nn.Linear(3, 1),
            train,
            val,
            trials=1,
            epochs=4,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            checkpoint=ckpt,
            verbose=False,
        )
        assert len(epochs_run) == 3  # epochs 2, 3, 4

    def test_reports_when_nothing_was_saved(self, monkeypatch, capsys):
        # Without checkpoint= there is nothing to resume from, and the run
        # must say so rather than implying progress was kept.
        train, val = _reg_loaders(n=16, batch_size=8)
        real_evaluate = fit_mod._evaluate

        def evaluate_then_preempt(model, val_loader, loss_fn, device, metric=None):
            preempt._handler(signal.SIGTERM, None)
            return real_evaluate(model, val_loader, loss_fn, device, metric)

        monkeypatch.setattr(fit_mod, "_evaluate", evaluate_then_preempt)
        preempt.watch()  # no checkpoint -> fit() would not arm it itself
        fit(
            nn.Linear(3, 1),
            train,
            val,
            trials=1,
            epochs=4,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            verbose=False,
        )
        out = capsys.readouterr().out
        assert "preemption signal" in out
        assert "progress is lost" in out

    def test_no_signal_means_no_behavior_change(self, tmp_path):
        train, val = _reg_loaders(n=16, batch_size=8)
        _, _, _ = fit(
            nn.Linear(3, 1),
            train,
            val,
            trials=1,
            epochs=2,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            checkpoint=str(tmp_path / "run.pt"),
            patience=10,
            verbose=False,
        )
        assert preempt.preempted() is False


class TestSearchResume:
    def test_resume_runs_only_the_remaining_trials(self, tmp_path):
        train, val = _cls_loaders()
        path = str(tmp_path / "study.log")

        def run():
            return tune(
                nn.Linear(4, 3),
                train,
                val,
                trials=3,
                epochs_per_trial=1,
                space=_SMALL_SPACE,
                storage=_journal_storage(path),
                study_name="s",
                resume=True,
                seed=0,
                verbose=False,
            )[2]

        assert len(run().trials) == 3
        assert len(run().trials) == 3  # not 6

    def test_without_resume_the_trials_stack_up(self, tmp_path):
        # The parallel search relies on this: each rank owns a share.
        train, val = _cls_loaders()
        path = str(tmp_path / "study.log")

        def run():
            return tune(
                nn.Linear(4, 3),
                train,
                val,
                trials=2,
                epochs_per_trial=1,
                space=_SMALL_SPACE,
                storage=_journal_storage(path),
                study_name="s",
                seed=0,
                verbose=False,
            )[2]

        assert len(run().trials) == 2
        assert len(run().trials) == 4

    def test_fit_journals_the_search_next_to_the_checkpoint(self, tmp_path):
        import os

        train, val = _reg_loaders(n=16, batch_size=8)
        ckpt = str(tmp_path / "run.pt")
        fit(
            nn.Linear(3, 1),
            train,
            val,
            trials=1,
            epochs=1,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            checkpoint=ckpt,
            verbose=False,
        )
        assert os.path.exists(ckpt + ".study")

    def test_no_checkpoint_leaves_no_journal_behind(self, tmp_path, monkeypatch):
        import os

        monkeypatch.chdir(tmp_path)
        train, val = _reg_loaders(n=16, batch_size=8)
        fit(
            nn.Linear(3, 1),
            train,
            val,
            trials=1,
            epochs=1,
            epochs_per_trial=1,
            space=_SMALL_SPACE,
            verbose=False,
        )
        assert not [f for f in os.listdir(tmp_path) if f.endswith(".study")]
