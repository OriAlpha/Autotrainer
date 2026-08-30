"""Tests for framework callbacks in autotrainer.callbacks."""

from autotrainer.callbacks import (
    AutotrainerCallback,
    AutotrainerHuggingFaceCallback,
    AutotrainerKerasCallback,
    AutotrainerLightningCallback,
    autotrainer_lightgbm_callback,
    autotrainer_xgboost_callback,
)
from autotrainer.summary import SummaryTracker


def test_autotrainer_callback_vanilla(tmp_path):
    summary = SummaryTracker(logs_dir=tmp_path, user="cb_user")
    cb = AutotrainerCallback(summary_tracker=summary)

    cb.on_train_begin(params={"lr": 0.001, "optimizer": "AdamW"})
    cb.on_step_end(step=1, loss=0.95)
    cb.on_epoch_end(epoch=1, train_loss=0.8, val_loss=0.7, val_acc=75.5)
    cb.on_train_end()

    assert summary.user == "cb_user"
    assert (tmp_path / summary.run_id / "run.json").exists()


def test_autotrainer_huggingface_callback(tmp_path):
    summary = SummaryTracker(logs_dir=tmp_path, user="hf_user")
    cb = AutotrainerHuggingFaceCallback(summary_tracker=summary)

    class MockArgs:
        def to_dict(self):
            return {"learning_rate": 5e-5, "per_device_train_batch_size": 16}

    cb.on_init_end(MockArgs(), None, None)
    cb.on_log(
        MockArgs(),
        None,
        None,
        logs={"loss": 0.6, "eval_loss": 0.5, "eval_accuracy": 0.88, "epoch": 1.0},
    )
    cb.on_train_end(MockArgs(), None, None)

    assert (tmp_path / summary.run_id / "run.json").exists()


def test_autotrainer_lightning_callback(tmp_path):
    summary = SummaryTracker(logs_dir=tmp_path, user="pl_user")
    cb = AutotrainerLightningCallback(summary_tracker=summary)

    class MockTrainer:
        max_epochs = 3
        precision = "16-mixed"
        current_epoch = 0
        callback_metrics = {"train_loss": 0.45, "val_loss": 0.38, "val_acc": 0.92}

    class MockModule:
        hparams = {"hidden_dim": 256}

    trainer = MockTrainer()
    pl_module = MockModule()

    cb.on_fit_start(trainer, pl_module)
    cb.on_train_epoch_end(trainer, pl_module)
    cb.on_fit_end(trainer, pl_module)

    assert (tmp_path / summary.run_id / "run.json").exists()


def test_autotrainer_keras_callback(tmp_path):
    summary = SummaryTracker(logs_dir=tmp_path, user="keras_user")
    cb = AutotrainerKerasCallback(log_batches=True, summary_tracker=summary)

    cb.on_train_batch_end(0, logs={"loss": 1.2})
    cb.on_epoch_end(0, logs={"loss": 0.9, "val_loss": 0.8, "val_accuracy": 0.75})
    cb.on_train_end()

    assert (tmp_path / summary.run_id / "run.json").exists()


def test_autotrainer_xgboost_lightgbm_callbacks():
    xgb_cb = autotrainer_xgboost_callback()
    assert callable(xgb_cb)

    lgb_cb = autotrainer_lightgbm_callback()
    assert callable(lgb_cb)
