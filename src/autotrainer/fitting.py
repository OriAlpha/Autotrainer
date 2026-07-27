"""One-call training (PyTorch): tune the recipe, then fully train the winner.

    model, params, study = autotrainer.fit(model, train_loader, val_loader)

Composes the two halves of autotrainer:

    1. TUNE  - Optuna search over the training recipe (short trials).
    2. TRAIN - full retrain of the winning recipe from the model's ORIGINAL
               initial weights, distributed via prepare(), with a
               warmup+cosine schedule, mixed precision, and early stopping
               on the validation loss.

Under `autotrainer run` with multiple processes, the search itself is
parallel: trials are split across the ranks and pulled from a shared
Optuna journal-file study, one trial per process on its own device. The
winning recipe (and the inferred loss) is broadcast so every rank trains
the exact same configuration. If phase 1 is much longer than phase 2 on
some ranks, raise the collective timeout with the AUTOTRAINER_TIMEOUT env
var (seconds).

The phase-1 search and checkpoint helpers live in ``_fit_search.py``; they
are re-exported here so existing ``from autotrainer.fitting import ...`` of
``_unwrap`` / ``_load_checkpoint`` / ``_journal_storage`` etc. keeps working.
"""

from __future__ import annotations

import copy
from typing import Any

from ._fit_search import (  # noqa: F401  (re-exported for back-compat)
    _CHECKPOINT_FORMAT,
    _journal_storage,
    _load_checkpoint,
    _parallel_search,
    _save_checkpoint,
    _sync_from_rank0,
    _unwrap,
)
from .augment import augment_batch
from .tuning import _evaluate, _rebuild_loader, tune


def fit(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    *,
    test_loader: Any = None,
    trials: int = 20,
    epochs: int = 20,
    epochs_per_trial: int = 3,
    space: dict[str, Any] | None = None,
    loss: str | None = None,
    patience: int = 5,
    min_delta: float = 0.0,
    checkpoint: str | None = None,
    study_storage: str | None = None,
    seed: int = 0,
    verbose: bool = True,
    lr_scaling: str = "auto",
) -> tuple[Any, dict[str, Any], Any]:
    """Search the training recipe, then fully train the winner.

    Phase 1 (tune): Optuna search over lr / weight decay / optimizer /
    batch size (or a custom ``space``) on short trials. Under DDP the
    trials are split across all ranks through a shared journal-file study
    (see ``study_storage``), one trial per process on its own device; the
    winning recipe is then broadcast so every rank trains the same
    configuration.

    Phase 2 (train): the winning recipe is retrained from the model's
    ORIGINAL initial weights - not from the best trial's weights, which
    only saw ``epochs_per_trial`` epochs - through ``prepare()`` (DDP +
    DistributedSampler when launched distributed), with a warmup+cosine
    schedule, mixed precision, and early stopping on the val loss. The
    weights from the best epoch are restored before returning.

    Args:
        model: a ``torch.nn.Module``; never mutated.
        train_loader: training DataLoader (batch size may be overridden by
            the winning recipe if ``batch_size`` is in the search space).
        val_loader: validation DataLoader; scores trials, drives early
            stopping, and selects the best epoch. Not sharded - under DDP
            every rank evaluates the full val set.
        test_loader: optional held-out DataLoader the search never sees.
            When given, the final (best-epoch) model is scored on it and the
            number is printed and stored on the study - an honest
            generalization estimate that guards against reading too much into
            a val loss the wide search has implicitly optimized against.
        trials: number of Optuna trials in phase 1.
        epochs: maximum full-training epochs in phase 2. This stays your
            ceiling even though phase 1 searches an ``epochs`` knob of its
            own: that searched value is a *trial* budget (bounded by
            ``epochs_per_trial``) used to pick the recipe, and phase 2 already
            decides its own length adaptively via ``patience``. It is recorded
            in ``best_params`` for reproducibility but does not shorten - or
            extend - the final retrain.
        epochs_per_trial: the per-trial epoch budget in phase 1, and the upper
            bound of the searched ``epochs`` knob.
        space: custom search space, as in :func:`autotrainer.tune`.
        loss: override the inferred loss; one of ``"cross_entropy"``,
            ``"bce"``, ``"mse"``, ``"huber"``. If ``None``, inferred.
        patience: stop phase 2 after this many epochs without val
            improvement.
        min_delta: minimum val-loss decrease that counts as improvement.
        checkpoint: path for preemption-safe checkpointing. Rank 0 writes
            the full training state (weights, optimizer, scheduler, recipe,
            early-stop counters) atomically after every phase-2 epoch. If
            the file already exists when fit() starts, the search is
            skipped and training RESUMES from it - so a requeued SLURM job
            rerunning the same script picks up where it died. Delete the
            file to start fresh.
        study_storage: path of the shared Optuna journal file used for the
            parallel search when launched distributed. Defaults to
            ``.autotrainer_study_<jobid>.log`` in the working directory;
            on multi-node runs it must live on a filesystem all nodes
            share (SLURM working directories normally are).
        seed: Optuna sampler seed for reproducibility.
        verbose: print tuning output and per-epoch val losses.
        lr_scaling: ``"auto"`` (default) applies the standard lr<->batch-size
            rule (linear for SGD, square-root for Adam-family) in both the
            search and the phase-2 retrain, so the two stay consistent;
            ``"none"`` disables it. See :func:`autotrainer.tune`.

    Returns:
        ``(model, best_params, study)``. The model is the plain module
        (never DDP-wrapped) carrying the best epoch's weights. Under DDP,
        ``study`` is ``None`` on every rank except rank 0. When a
        ``test_loader`` was given, rank 0's ``study.user_attrs["test_loss"]``
        holds the held-out estimate.
    """
    import torch

    from .auto_optim import _infer_loss, _make_loss, _make_optimizer, _make_scheduler, _scale_lr
    from .backends.torch_backend import _ensure_process_group, prepare
    from .utils import (
        GradScaler,
        autocast_context,
        print0,
        rank,
        robust_forward,
        set_epoch,
        split_xy,
        to_device,
    )

    if not hasattr(model, "state_dict"):
        raise TypeError(
            f"fit() currently supports PyTorch models, got {type(model)!r}. "
            "For sklearn/XGBoost/LightGBM estimators use autotrainer.tune()."
        )

    distributed = _ensure_process_group()
    init_state = copy.deepcopy(model.state_dict())

    # Resume: an existing checkpoint carries the winning recipe and the full
    # training state, so the search is skipped entirely. Every rank reads
    # the same file, so no broadcast is needed for the recipe.
    ckpt = _load_checkpoint(checkpoint)
    if ckpt is not None and verbose:
        print0(
            f"[autotrainer] fit: resuming from {checkpoint} "
            f"(epoch {ckpt['epoch'] + 1} done, best val_loss={ckpt['best_val']:.4f})"
        )

    # ---- Phase 1: search the recipe ----
    study = None
    best_params: dict[str, Any] = {}
    if ckpt is not None:
        # A resumed run already agrees on the recipe - all ranks read the file.
        best_params, loss = ckpt["params"], ckpt["loss"]
    else:
        if loss is None and (not distributed or rank() == 0):
            xb, yb = split_xy(next(iter(train_loader)))
            if yb is None:
                raise ValueError(
                    "autotrainer.fit could not find targets in your batches to "
                    "infer the loss. Pass loss=... explicitly, e.g. "
                    "fit(model, train_loader, val_loader, loss='mse')."
                )
            _, loss, loss_why = _infer_loss(model, yb, xb)
            if verbose:
                print(f"[autotrainer] fit: loss={loss} ({loss_why})")
        if distributed:
            # Everyone must search with the SAME loss - a shuffled first
            # batch could otherwise flip the inference on some rank.
            [loss] = _sync_from_rank0([loss], True)
        assert loss is not None  # inferred above or user-provided

        if not distributed:
            _, best_params, study = tune(
                model,
                train_loader,
                val_loader,
                trials=trials,
                epochs_per_trial=epochs_per_trial,
                space=space,
                loss=loss,
                seed=seed,
                verbose=verbose,
                lr_scaling=lr_scaling,
            )
        else:
            import os

            key = os.environ.get("SLURM_JOB_ID") or os.environ.get("MASTER_PORT", "29500")
            best_params, study = _parallel_search(
                model,
                train_loader,
                val_loader,
                trials=trials,
                epochs_per_trial=epochs_per_trial,
                space=space,
                loss=loss,
                seed=seed,
                verbose=verbose,
                lr_scaling=lr_scaling,
                storage_path=study_storage or f".autotrainer_study_{key}.log",
            )
            # Ranks > 0 hold an empty dict; rank 0 read the winner.
            [best_params] = _sync_from_rank0([best_params], True)
    # Train with the winner's label_smoothing (if it was searched); SCORE with
    # the plain unsmoothed loss so the val/test numbers and the early-stop
    # decision stay honest and comparable across configs.
    loss_fn = _make_loss(loss, label_smoothing=best_params.get("label_smoothing", 0.0))
    eval_loss_fn = _make_loss(loss)

    # ---- Phase 2: full retrain of the winner from the original init ----
    m = copy.deepcopy(model)
    m.load_state_dict(init_state)
    tl = (
        _rebuild_loader(train_loader, best_params["batch_size"])
        if "batch_size" in best_params
        else train_loader
    )
    m, tl = prepare(m, tl)
    device = next(m.parameters()).device
    lr_applied = _scale_lr(
        best_params.get("lr", 1e-3),
        best_params.get("batch_size"),
        best_params.get("optimizer"),
        lr_scaling,
    )
    opt, opt_name, _ = _make_optimizer(
        m,
        best_params.get("optimizer"),
        lr_applied,
        best_params.get("weight_decay", 0.0),
    )

    steps = max(len(tl) * epochs, 1)
    sched_name = best_params.get("scheduler", "cosine")
    sched = _make_scheduler(sched_name, opt, steps, best_params.get("warmup_frac", 0.05))
    grad_clip = best_params.get("grad_clip", 0.0)
    # The winning augmentation strength is part of the recipe that earned the
    # best val score, so the full retrain has to train under it too - dropping
    # it here would retrain a *different*, unregularized recipe.
    aug_strength = best_params.get("aug_strength", 0.0)
    if verbose:
        print0(
            f"[autotrainer] fit: retraining winner from original init "
            f"(optimizer={opt_name}, schedule={sched_name}, up to {epochs} epochs, "
            f"patience={patience})"
        )

    scaler = GradScaler()
    best_val, best_state, bad_epochs, start_epoch = float("inf"), None, 0, 0
    if ckpt is not None:
        _unwrap(m).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if sched is not None and ckpt.get("scheduler") is not None:
            sched.load_state_dict(ckpt["scheduler"])
        best_val = ckpt["best_val"]
        best_state = ckpt["best_state"]
        bad_epochs = ckpt["bad_epochs"]
        start_epoch = ckpt["epoch"] + 1
        if bad_epochs >= patience:
            start_epoch = epochs  # run had already early-stopped; nothing left to train

    for epoch in range(start_epoch, epochs):
        set_epoch(tl, epoch)
        m.train()
        for batch in tl:
            xb, yb = split_xy(batch)
            # Train-path only; _evaluate() below sees clean inputs.
            xb_dev = augment_batch(to_device(xb, device), aug_strength)
            yb_dev = to_device(yb, device)
            opt.zero_grad()
            with autocast_context():
                out = robust_forward(m, xb_dev)
                loss_val = loss_fn(out, yb_dev)
            scaler.scale(loss_val).backward()
            if grad_clip:
                scaler.unscale_(opt)  # unscale before clipping so the norm is real
                torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()

        val = _evaluate(m, val_loader, eval_loss_fn, device)
        if distributed:
            # Every rank computes the same val loss up to float rounding,
            # but the early-stop decision must be bit-identical everywhere
            # or the ranks desynchronize - so rank 0's number wins.
            import torch.distributed as dist

            t = torch.tensor([val], dtype=torch.float64, device=device)
            dist.broadcast(t, src=0)
            val = float(t.item())

        improved = val < best_val - min_delta
        if verbose:
            print0(
                f"[autotrainer] fit: epoch {epoch + 1}/{epochs} "
                f"val_loss={val:.4f}{' *' if improved else ''}"
            )
        if improved:
            best_val, bad_epochs = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in _unwrap(m).state_dict().items()}
        else:
            bad_epochs += 1

        if checkpoint is not None:
            _save_checkpoint(
                checkpoint,
                {
                    "format_version": _CHECKPOINT_FORMAT,
                    "params": best_params,
                    "loss": loss,
                    "epoch": epoch,
                    "model": {
                        k: v.detach().cpu().clone() for k, v in _unwrap(m).state_dict().items()
                    },
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict() if sched is not None else None,
                    "best_val": best_val,
                    "best_state": best_state,
                    "bad_epochs": bad_epochs,
                },
            )

        if bad_epochs >= patience:
            if verbose:
                print0(
                    f"[autotrainer] fit: early stop at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
            break

    final = _unwrap(m)
    if best_state is not None:
        final.load_state_dict(best_state)
    if verbose:
        print0(f"[autotrainer] fit: done - best val_loss={best_val:.4f} with {best_params}")

    # An honest generalization number: val_loss drove selection, so the more
    # the search widened, the more it can flatter the val set. A held-out test
    # set the search never saw is the number to actually trust.
    if test_loader is not None:
        test_loss = _evaluate(final, test_loader, eval_loss_fn, device)
        if study is not None:  # None on ranks > 0 and on a resumed (search-skipped) run
            study.set_user_attr("test_loss", test_loss)
        if verbose:
            print0(
                f"[autotrainer] fit: held-out test_loss={test_loss:.4f} "
                f"(val_loss={best_val:.4f} guided selection; test is the honest estimate)"
            )
    return final, best_params, study
