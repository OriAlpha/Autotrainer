"""Hyperparameter tuning (PyTorch): the model is the user's, the recipe is ours.

The user supplies a built model; we search over TRAINING hyperparameters
only (lr, weight decay, optimizer, batch size). No architecture search.

    best_model, best_params, study = autotrainer.tune(
        model, train_loader, val_loader, trials=30
    )

Every trial starts from the model's ORIGINAL initial weights (deep-copied),
so trials are comparable and the user's model object is never mutated.
Bad trials are pruned early by ASHA (successive halving) to save compute, so
widening the search space stays affordable.
"""

from __future__ import annotations

import copy
from typing import Any

# The maximal recipe space: every knob tune()/fit() can search. The actual
# default is narrowed per task by _default_space(); this superset stays so
# callers (and tests) can reason about "all possible" search keys.
DEFAULT_SPACE = {
    "lr": ("loguniform", 1e-5, 1e-1),
    "weight_decay": ("loguniform", 1e-6, 1e-1),
    "optimizer": ("categorical", ["adamw", "sgd"]),
    "batch_size": ("categorical", [16, 32, 64, 128]),
    "scheduler": ("categorical", ["cosine", "onecycle", "constant"]),
    "warmup_frac": ("uniform", 0.0, 0.1),
    "grad_clip": ("categorical", [0.0, 1.0, 5.0]),
    "label_smoothing": ("uniform", 0.0, 0.1),
}


def _default_space(model: Any, loss_name: str) -> dict[str, Any]:
    """Task-aware default search space (a subset of ``DEFAULT_SPACE``'s keys).

    A good default *narrows* the maximal space to the model and task so trials
    are spent where they pay off:

      * CNNs (conv layers present) lean on the classic SGD + 1cycle recipe and
        tolerate higher LRs; everything else defaults to a lower LR band with a
        cosine/constant schedule.
      * ``label_smoothing`` is only searched for cross-entropy - it's a
        classification-only regularizer and a wasted dimension elsewhere.

    ``grad_clip=0.0`` means "no clipping", so the "off" case is always in the
    space rather than being unreachable.
    """
    from .auto_optim import _looks_like_cnn

    is_cnn = _looks_like_cnn(model)
    space: dict[str, Any] = {
        "lr": ("loguniform", 1e-4, 5e-1) if is_cnn else ("loguniform", 1e-5, 1e-2),
        "weight_decay": ("loguniform", 1e-6, 1e-1),
        "optimizer": ("categorical", ["sgd", "adamw"]),
        "batch_size": ("categorical", [16, 32, 64, 128]),
        "scheduler": (
            ("categorical", ["onecycle", "cosine"])
            if is_cnn
            else ("categorical", ["cosine", "constant"])
        ),
        "warmup_frac": ("uniform", 0.0, 0.1),
        "grad_clip": ("categorical", [0.0, 1.0, 5.0]),
    }
    if loss_name == "cross_entropy":
        space["label_smoothing"] = ("uniform", 0.0, 0.1)
    return space


def _suggest(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
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

    from .utils import get_batch_size, robust_forward, split_xy, to_device

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            xb, yb = split_xy(batch)
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
    space: dict[str, Any] | None = None,
    loss: str | None = None,
    seed: int = 0,
    verbose: bool = True,
    storage: Any = None,
    study_name: str | None = None,
    pruner: Any = None,
    lr_scaling: str = "auto",
) -> tuple[Any, dict[str, Any], Any]:
    """Search training hyperparameters for the user's model.

    Searches over the training *recipe* only (lr, weight decay, optimizer,
    batch size, LR schedule, warmup, gradient clipping, and - for
    classification - label smoothing) - never the architecture. Every trial
    starts from the model's ORIGINAL initial weights (deep-copied), so trials
    are comparable and the input model is left untouched. Bad trials are pruned
    early by ASHA (successive halving) so a wide search stays affordable - most
    candidates get a small budget and only the survivors are promoted.

    Args:
        model: a ``torch.nn.Module``; never mutated.
        train_loader: training DataLoader (batch size may be overridden
            per trial if ``batch_size`` is in the search space).
        val_loader: validation DataLoader used to score each trial.
        trials: number of Optuna trials to run.
        epochs_per_trial: epochs trained per trial before scoring/pruning.
        space: custom search space; defaults to a task-aware subset of
            ``DEFAULT_SPACE`` (chosen from the model + inferred loss). Each
            entry is ``(kind, *args)`` where kind is one of
            ``loguniform``/``uniform``/``int``/``categorical``.
        loss: override the inferred loss; if ``None``, inferred once from
            the first training batch.
        seed: Optuna TPESampler seed for reproducibility.
        verbose: print the inferred loss and a final summary.
        storage: an Optuna storage (e.g. ``JournalStorage``) shared by
            several processes so they pull trials from one study - this is
            how ``fit()`` runs one trial per rank. With a storage, the
            study is created with ``load_if_exists=True``.
        study_name: name of the (shared) study; required to rejoin an
            existing study in ``storage``.
        pruner: an Optuna pruner; defaults to ASHA
            (``SuccessiveHalvingPruner``), the multi-fidelity strategy that
            gives most trials a small budget and promotes only survivors so a
            wide space stays cheap. Pass your own to override. ``min_resource``
            is 1, so trials are never pruned before their second epoch.
        lr_scaling: ``"auto"`` (default) applies the standard lr<->batch-size
            rule when both are searched - linear for SGD, square-root for
            Adam-family - relative to a reference batch of 32, so the search
            doesn't waste trials rediscovering that coupling. ``"none"``
            disables it. ``best_params`` always records the un-scaled searched
            lr; the scaling is applied when the optimizer is built.

    Returns:
        ``(best_model, best_params, study)`` where ``best_model`` carries the
        weights from the best trial.
    """
    import optuna
    import torch

    from .auto_optim import _infer_loss, _make_loss, _make_optimizer, _make_scheduler, _scale_lr
    from .utils import cuda_device, split_xy

    device = cuda_device()
    init_state = copy.deepcopy(model.state_dict())

    # Infer the loss NAME once, on the untouched model (or take the override).
    # The name drives both the eval criterion and the task-aware default space.
    xb, yb = split_xy(next(iter(train_loader)))
    if loss is not None:
        loss_name = loss
    elif yb is None:
        raise ValueError(
            "autotrainer.tune could not find targets in your batches to infer "
            "the loss. Pass loss=... explicitly, e.g. "
            "tune(model, train_loader, val_loader, loss='mse')."
        )
    else:
        _, loss_name, why = _infer_loss(model, yb, xb)
        if verbose:
            print(f"[autotrainer] tune: loss={loss_name} ({why})")

    space = space or _default_space(model, loss_name)
    # Trials are SCORED with a fixed, unsmoothed loss so their val numbers stay
    # comparable when label_smoothing is being searched (smoothing raises the
    # loss floor, which would otherwise make smoothed trials look artificially
    # worse than they are).
    eval_loss_fn = _make_loss(loss_name)

    best: dict[str, Any] = {"loss": float("inf"), "state": None}

    def objective(trial: Any) -> float:
        params = _suggest(trial, space)
        m = copy.deepcopy(model).to(device)
        m.load_state_dict(init_state)  # every trial starts identically
        lr_applied = _scale_lr(
            params.get("lr", 1e-3), params.get("batch_size"), params.get("optimizer"), lr_scaling
        )
        opt, _, _ = _make_optimizer(
            m,
            params.get("optimizer"),
            lr_applied,
            params.get("weight_decay", 0.0),
        )
        tl = (
            _rebuild_loader(train_loader, params["batch_size"])
            if "batch_size" in params
            else train_loader
        )
        train_loss_fn = _make_loss(loss_name, label_smoothing=params.get("label_smoothing", 0.0))
        total_steps = max(len(tl) * epochs_per_trial, 1)
        sched = _make_scheduler(
            params.get("scheduler", "cosine"), opt, total_steps, params.get("warmup_frac", 0.05)
        )
        grad_clip = params.get("grad_clip", 0.0)

        from .utils import robust_forward, split_xy, to_device

        val = float("inf")
        for epoch in range(epochs_per_trial):
            m.train()
            for batch in tl:
                bx, by = split_xy(batch)
                bx_dev = to_device(bx, device)
                by_dev = to_device(by, device)
                opt.zero_grad()
                out = robust_forward(m, bx_dev)
                loss_val = train_loss_fn(out, by_dev)
                loss_val.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
                opt.step()
                if sched is not None:
                    sched.step()
            val = _evaluate(m, val_loader, eval_loss_fn, device)
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
            pruner=pruner
            or optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3),
            storage=storage,
            study_name=study_name,
            load_if_exists=storage is not None,
        )
        study.optimize(objective, n_trials=trials)
    finally:
        optuna.logging.set_verbosity(prior_verbosity)

    best_model = copy.deepcopy(model)
    if best["state"] is not None:
        best_model.load_state_dict(best["state"])

    # A zero-trial share (parallel search with more ranks than trials) or a
    # racing shared study may have no completed trial visible yet.
    try:
        best_params, best_value = study.best_params, study.best_value
    except ValueError:
        best_params, best_value = {}, float("nan")

    if verbose:
        pruned = sum(t.state.name == "PRUNED" for t in study.trials)
        print(
            f"[autotrainer] tune: best val loss {best_value:.4f} "
            f"with {best_params} ({pruned}/{trials} trials pruned early)"
        )
    return best_model, best_params, study
