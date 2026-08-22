"""Framework integration callbacks for autotrainer experiment tracking.

Provides native callbacks for:
- PyTorch / Vanilla Python loops (`AutotrainerCallback`)
- Hugging Face Transformers (`AutotrainerHuggingFaceCallback`)
- PyTorch Lightning (`AutotrainerLightningCallback`)
- Keras / TensorFlow (`AutotrainerKerasCallback`)
- XGBoost (`autotrainer_xgboost_callback`)
- LightGBM (`autotrainer_lightgbm_callback`)

All callbacks seamlessly stream metrics, parameters, and health triage data
into autotrainer's active summary tracker for real-time Web UI rendering.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from .summary import SummaryTracker, get_active_summary


class AutotrainerCallback:
    """General-purpose training callback for autotrainer.

    Can be used directly in custom training loops or subclassed for custom frameworks.
    """

    def __init__(self, summary_tracker: SummaryTracker | None = None) -> None:
        self._summary = summary_tracker

    @property
    def summary(self) -> SummaryTracker:
        return self._summary or get_active_summary()

    def on_train_begin(self, params: dict[str, Any] | None = None) -> None:
        if params:
            self.summary.log_params(params)

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
        val_acc: float | None = None,
        **extra_metrics: Any,
    ) -> None:
        self.summary.log_epoch(
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            **extra_metrics,
        )

    def on_step_end(self, step: int, loss: float | None = None, **extra_metrics: Any) -> None:
        self.summary.step(loss=loss, **extra_metrics)

    def on_train_end(self, summary_data: dict[str, Any] | None = None) -> None:
        self.summary.end(summary_data=summary_data)


class AutotrainerHuggingFaceCallback:
    """Hugging Face Transformers TrainerCallback for automatic autotrainer logging."""

    def __init__(self, summary_tracker: SummaryTracker | None = None) -> None:
        self._summary = summary_tracker

    @property
    def summary(self) -> SummaryTracker:
        return self._summary or get_active_summary()

    def on_init_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if hasattr(args, "to_dict"):
            with contextlib.suppress(Exception):
                self.summary.log_params(args.to_dict())

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        if not logs:
            return

        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        eval_acc = logs.get("eval_accuracy") or logs.get("eval_acc")
        epoch = logs.get("epoch")

        if eval_acc is not None and eval_acc <= 1.0:
            eval_acc = eval_acc * 100.0

        if epoch is not None:
            int_epoch = int(round(epoch))
            self.summary.log_epoch(
                epoch=int_epoch,
                train_loss=float(loss) if loss is not None else None,
                val_loss=float(eval_loss) if eval_loss is not None else None,
                val_acc=float(eval_acc) if eval_acc is not None else None,
            )
        elif loss is not None:
            self.summary.step(loss=float(loss))

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.summary.end()


class AutotrainerLightningCallback:
    """PyTorch Lightning Callback for automatic autotrainer metric logging."""

    def __init__(self, summary_tracker: SummaryTracker | None = None) -> None:
        self._summary = summary_tracker

    @property
    def summary(self) -> SummaryTracker:
        return self._summary or get_active_summary()

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        params: dict[str, Any] = {}
        if hasattr(trainer, "max_epochs"):
            params["max_epochs"] = trainer.max_epochs
        if hasattr(trainer, "precision"):
            params["precision"] = str(trainer.precision)
        if hasattr(pl_module, "hparams"):
            with contextlib.suppress(Exception):
                params.update(dict(pl_module.hparams))
        if params:
            self.summary.log_params(params)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        cb_metrics = getattr(trainer, "callback_metrics", {})
        epoch = getattr(trainer, "current_epoch", 0) + 1

        def _get_metric(keys: list[str]) -> float | None:
            for k in keys:
                if k in cb_metrics:
                    val = cb_metrics[k]
                    try:
                        return float(val.item() if hasattr(val, "item") else val)
                    except Exception:
                        pass
            return None

        train_loss = _get_metric(["train_loss", "loss", "train/loss"])
        val_loss = _get_metric(["val_loss", "val/loss", "eval_loss"])
        val_acc = _get_metric(["val_acc", "val_accuracy", "val/acc", "val/accuracy"])
        if val_acc is not None and val_acc <= 1.0:
            val_acc = val_acc * 100.0

        self.summary.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
        )

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        self.summary.end()


class AutotrainerKerasCallback:
    """Keras / TensorFlow Callback for autotrainer metric reporting and Web UI tracking."""

    def __init__(
        self, log_batches: bool = False, summary_tracker: SummaryTracker | None = None
    ) -> None:
        self.log_batches = log_batches
        self._summary = summary_tracker
        self.step_count = 0

    @property
    def summary(self) -> SummaryTracker:
        return self._summary or get_active_summary()

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        if self.log_batches and logs:
            self.step_count += 1
            loss = logs.get("loss")
            if loss is not None:
                self.summary.step(loss=float(loss))

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs:
            train_loss = logs.get("loss", 0.0)
            val_loss = logs.get("val_loss")
            val_acc = logs.get("val_accuracy") or logs.get("val_acc")
            if val_acc is not None and val_acc <= 1.0:
                val_acc = val_acc * 100.0
            self.summary.log_epoch(
                epoch=epoch + 1,
                train_loss=float(train_loss),
                val_loss=float(val_loss) if val_loss is not None else None,
                val_acc=float(val_acc) if val_acc is not None else None,
            )

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.summary.end()


def autotrainer_xgboost_callback() -> Callable[[Any], None]:
    """Returns an XGBoost callback function for logging per-iteration evaluation metrics."""

    def callback(env: Any) -> None:
        evals_result = getattr(env, "evaluation_result_list", [])
        if evals_result:
            train_loss = None
            val_loss = None
            for item in evals_result:
                name, val = item[0], item[1]
                if "train" in name and train_loss is None:
                    train_loss = float(val)
                elif ("val" in name or "eval" in name or "test" in name) and val_loss is None:
                    val_loss = float(val)
            if train_loss is not None:
                get_active_summary().log_epoch(train_loss=train_loss, val_loss=val_loss)

    return callback


def autotrainer_lightgbm_callback() -> Callable[[Any], None]:
    """Returns a LightGBM callback function for logging per-iteration evaluation metrics."""

    def callback(env: Any) -> None:
        evals = getattr(env, "evaluation_result_list", [])
        if evals:
            train_loss = None
            val_loss = None
            for data_name, _eval_name, val, _ in evals:
                if data_name == "train" and train_loss is None:
                    train_loss = float(val)
                elif data_name in ("val", "valid", "test", "eval") and val_loss is None:
                    val_loss = float(val)
            if train_loss is not None:
                get_active_summary().log_epoch(train_loss=train_loss, val_loss=val_loss)

    return callback
