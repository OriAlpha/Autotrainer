"""One-call training (all frameworks):

    model = autotrainer.train(model, loader, epochs=5, save_path="model.pt")

Infers the recipe via :func:`autotrainer.auto` (loss, optimizer, LR,
schedule), distributes via ``prepare()``, runs the epochs, prints the
summary, and saves. PyTorch, Keras, sklearn-API estimators and the native
XGBoost API all route from here.
"""

from __future__ import annotations

from typing import Any


def train(
    model: Any,
    loader: Any = None,
    y: Any = None,
    *,
    epochs: int = 10,
    lr: float | None = None,
    loss_fn: Any | None = None,
    optimizer: Any | None = None,
    patience: int | None = None,
    save_path: str | Any | None = None,
) -> Any:
    """One-line complete training loop.

    Infers loss function, optimizer, LR schedule, and mixed precision scaling,
    runs all training epochs, prints the performance summary, optionally saves the model,
    and returns the trained model. Supports PyTorch, Scikit-Learn, and XGBoost models.

    Usage:
        # PyTorch:
        model = autotrainer.train(model, loader, epochs=5, save_path="model.pt")

        # Scikit-Learn / XGBoost:
        search = autotrainer.train(search, X, y)

    Note ``epochs`` means passes over the data for PyTorch and Keras, and
    ``num_boost_round`` for the native-XGBoost path (a params dict + DMatrix),
    where there are no epochs to make passes over.
    """
    from .utils import framework_of

    # Routed the same way as prepare() and tune(): by module prefix with an
    # isinstance fallback, rather than by probing for `.fit` / `.forward`
    # attributes - duck-typing here misroutes anything that happens to define
    # a fit() method, and silently picked the sklearn path for it.
    if isinstance(model, dict):
        # Native XGBoost API: a params dict plus a DMatrix, not an estimator.
        if loader is None:
            raise TypeError(
                "train(): a params dict is the native-XGBoost path and needs a "
                "DMatrix as the second argument."
            )
        import xgboost as xgb

        from .backends.boosting_backend import boost_params

        params = boost_params(model)
        booster = xgb.train(params, loader, num_boost_round=epochs)
        if save_path is not None:
            booster.save_model(save_path)
            from .utils import print0

            print0(f"[autotrainer] saved XGBoost model to {save_path}")
        from .summary import finish

        finish(checkpoint=save_path)
        return booster

    framework = framework_of(model)

    if framework == "tf":
        import tensorflow as tf

        from .backends.tf_backend import scale_batch_size

        bs = scale_batch_size(64)
        callbacks = []
        if patience is not None:
            es = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
            callbacks.append(es)
        if loader is not None:
            if y is not None:
                model.fit(loader, y, batch_size=bs, epochs=epochs, callbacks=callbacks)
            else:
                model.fit(loader, epochs=epochs, callbacks=callbacks)
        if save_path is not None:
            model.save(save_path)
            from .utils import print0

            print0(f"[autotrainer] saved TensorFlow model to {save_path}")
        from .summary import finish

        finish(checkpoint=save_path)
        return model

    if framework in ("sklearn", "boosting"):
        # Boosting estimators get the boosting backend's thread config rather
        # than joblib's n_jobs handling, matching how prepare() routes them.
        if framework == "boosting":
            from .backends.boosting_backend import prepare as estimator_prepare
        else:
            from .backends.sklearn_backend import prepare as estimator_prepare

        estimator = estimator_prepare(model)
        if loader is not None:
            if y is not None:
                estimator.fit(loader, y)
            else:
                estimator.fit(loader)
        if save_path is not None:
            import joblib

            joblib.dump(estimator, save_path)
            from .utils import print0

            print0(f"[autotrainer] saved estimator to {save_path}")
        from .summary import finish

        finish(checkpoint=save_path)
        return estimator

    if framework != "torch":
        raise TypeError(
            f"train() supports PyTorch modules, Keras models, sklearn-API "
            f"estimators (incl. XGBoost/LightGBM), and native-XGBoost params "
            f"dicts; got {type(model)!r}."
        )

    from .auto_optim import auto
    from .summary import finish, get_active_summary
    from .utils import print0, save0

    model, loader, opt, loss_fn, sched = auto(
        model, loader, epochs=epochs, lr=lr, loss=loss_fn, optimizer=optimizer
    )

    device = next(model.parameters()).device
    summary = get_active_summary()
    summary.batch_size = getattr(loader, "batch_size", None)
    summary.optimizer = opt
    summary.loss_fn = loss_fn
    # auto() infers the schedule; without this the summary's "LR Schedule"
    # line stays blank on the one path that always has a scheduler.
    summary.scheduler = sched

    import torch

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    out = model(xb)
                    loss = loss_fn(out, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
            # One .item() per step, reused. Each call is a device sync, so
            # reading it twice stalled the pipeline twice per iteration - in
            # the entry point whose whole selling point is throughput.
            loss_value = loss.item()
            total_loss += loss_value
            summary.step(loss=loss_value)
        if sched is not None:
            sched.step()
        epoch_loss = total_loss / max(len(loader), 1)
        summary.log_epoch(train_loss=epoch_loss)
        print0(f"epoch {epoch + 1}/{epochs}: loss {epoch_loss:.4f}")

    if save_path is not None:
        save0(model.state_dict(), save_path)

    finish(checkpoint=save_path)
    return model
