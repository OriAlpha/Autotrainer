"""Hyperparameter tuning (PyTorch): the model is the user's, the recipe is ours.

The user supplies a built model; we search over TRAINING hyperparameters
only (lr, weight decay, optimizer, batch size). No architecture search.

    best_model, best_params, study = autotrainer.tune(
        model, train_loader, val_loader, trials=30
    )

Every trial starts from the model's ORIGINAL initial weights (deep-copied),
so trials are comparable and the user's model object is never mutated.
Bad trials are pruned early by Optuna's median pruner to save compute.
"""

from __future__ import annotations

import copy
from typing import Any

DEFAULT_SPACE = {
    "lr": ("loguniform", 1e-5, 1e-1),
    "weight_decay": ("loguniform", 1e-6, 1e-1),
    "optimizer": ("categorical", ["adamw", "sgd"]),
    "batch_size": ("categorical", [16, 32, 64, 128]),
}


def _suggest(trial: Any, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "loguniform":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "uniform":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown space kind {kind!r} for {name!r}")
    return params


def _rebuild_loader(loader: Any, batch_size: int) -> Any:
    from torch.utils.data import DataLoader

    from .backends.torch_backend import _loader_kwargs

    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        **_loader_kwargs(loader),
    )


def _evaluate(model: Any, val_loader: Any, loss_fn: Any, device: Any) -> float:
    import torch

    from .utils import get_batch_size, robust_forward, to_device

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb_dev = to_device(xb, device)
            yb_dev = to_device(yb, device)
            out = robust_forward(model, xb_dev)
            bs = get_batch_size(yb_dev) or get_batch_size(xb_dev) or 1
            total += loss_fn(out, yb_dev).item() * bs
            n += bs
    return total / max(n, 1)


def tune(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    *,
    trials: int = 20,
    epochs_per_trial: int = 3,
    space: dict | None = None,
    loss: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[Any, dict, Any]:
    """Search training hyperparameters for the user's model.

    Searches over the training *recipe* only (lr, weight decay, optimizer,
    batch size) - never the architecture. Every trial starts from the model's
    ORIGINAL initial weights (deep-copied), so trials are comparable and the
    input model is left untouched. Bad trials are pruned early by Optuna's
    median pruner to save compute.

    Args:
        model: a ``torch.nn.Module``; never mutated.
        train_loader: training DataLoader (batch size may be overridden
            per trial if ``batch_size`` is in the search space).
        val_loader: validation DataLoader used to score each trial.
        trials: number of Optuna trials to run.
        epochs_per_trial: epochs trained per trial before scoring/pruning.
        space: custom search space (defaults to ``DEFAULT_SPACE``). Each
            entry is ``(kind, *args)`` where kind is one of
            ``loguniform``/``uniform``/``int``/``categorical``.
        loss: override the inferred loss; if ``None``, inferred once from
            the first training batch.
        seed: Optuna TPESampler seed for reproducibility.
        verbose: print the inferred loss and a final summary.

    Returns:
        ``(best_model, best_params, study)`` where ``best_model`` carries the
        weights from the best trial.
    """
    import optuna
    import torch

    from .auto_optim import _infer_loss, _make_loss, _make_optimizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    space = space or DEFAULT_SPACE
    init_state = copy.deepcopy(model.state_dict())

    # Infer the loss once, on the untouched model (or take the override).
    xb, yb = next(iter(train_loader))
    if loss is not None:
        loss_fn = _make_loss(loss)
    else:
        loss_fn, loss_name, why = _infer_loss(model, yb, xb)
        if verbose:
            print(f"[autotrainer] tune: loss={loss_name} ({why})")

    best: dict[str, Any] = {"loss": float("inf"), "state": None}

    def objective(trial):
        params = _suggest(trial, space)
        m = copy.deepcopy(model).to(device)
        m.load_state_dict(init_state)  # every trial starts identically
        opt, _, _ = _make_optimizer(
            m,
            params.get("optimizer"),
            params.get("lr", 1e-3),
            params.get("weight_decay", 0.0),
        )
        tl = (
            _rebuild_loader(train_loader, params["batch_size"])
            if "batch_size" in params
            else train_loader
        )

        from .utils import robust_forward, to_device

        for epoch in range(epochs_per_trial):
            m.train()
            for bx, by in tl:
                bx_dev = to_device(bx, device)
                by_dev = to_device(by, device)
                opt.zero_grad()
                out = robust_forward(m, bx_dev)
                loss_val = loss_fn(out, by_dev)
                loss_val.backward()
                opt.step()
            val = _evaluate(m, val_loader, loss_fn, device)
            trial.report(val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val < best["loss"]:
            best["loss"] = val
            best["state"] = copy.deepcopy(m.state_dict())
        return val

    # Optuna's verbosity is process-global; save and restore it so a tune()
    # call doesn't quietly change logging behavior for the rest of the program.
    prior_verbosity = optuna.logging.get_verbosity()
    try:
        optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=trials)
    finally:
        optuna.logging.set_verbosity(prior_verbosity)

    best_model = copy.deepcopy(model)
    if best["state"] is not None:
        best_model.load_state_dict(best["state"])

    if verbose:
        pruned = sum(t.state.name == "PRUNED" for t in study.trials)
        print(
            f"[autotrainer] tune: best val loss {study.best_value:.4f} "
            f"with {study.best_params} ({pruned}/{trials} trials pruned early)"
        )
    return best_model, study.best_params, study
