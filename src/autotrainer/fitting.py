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
"""

from __future__ import annotations

import copy
from typing import Any

from .tuning import _evaluate, _rebuild_loader, tune


def _unwrap(model: Any) -> Any:
    from torch.nn.parallel import DistributedDataParallel as DDP

    return model.module if isinstance(model, DDP) else model


def _sync_from_rank0(payload: list[Any], distributed: bool) -> list[Any]:
    """Broadcast a picklable payload from rank 0 (no-op when not distributed)."""
    if distributed:
        import torch.distributed as dist

        dist.broadcast_object_list(payload, src=0)
    return payload


def _journal_storage(path: str) -> Any:
    """File-based Optuna storage that is safe on shared/NFS filesystems.

    The open()-based lock replaces the default symlink lock, which does not
    work on Windows and is unreliable on some NFS mounts.
    """
    from optuna.storages import JournalStorage

    try:  # optuna >= 4
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

        return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    except ImportError:  # optuna 3.x
        from optuna.storages import JournalFileOpenLock, JournalFileStorage

        return JournalStorage(JournalFileStorage(path, lock_obj=JournalFileOpenLock(path)))


def _parallel_search(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    *,
    trials: int,
    epochs_per_trial: int,
    space: dict[str, Any] | None,
    loss: str,
    seed: int,
    verbose: bool,
    storage_path: str,
) -> tuple[dict[str, Any], Any]:
    """One search, every rank working: trials are split across the ranks and
    pulled from a shared journal-file study, so the whole allocation is busy
    during phase 1 instead of idling behind rank 0. Each rank trains its
    trials on its own device (LOCAL_RANK); samplers are seeded per rank so
    the ranks propose different candidates."""
    import contextlib
    import os

    from .utils import barrier

    r = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if r == 0:  # a stale study from a previous run must not pollute this one
        for stale in (storage_path, storage_path + ".lock"):
            with contextlib.suppress(OSError):
                os.unlink(stale)
    barrier()

    share = trials // world_size + (1 if r < trials % world_size else 0)
    _, _, study = tune(
        model,
        train_loader,
        val_loader,
        trials=share,
        epochs_per_trial=epochs_per_trial,
        space=space,
        loss=loss,
        seed=seed + r,
        verbose=verbose and r == 0,
        storage=_journal_storage(storage_path),
        study_name="autotrainer-fit",
    )
    barrier()  # every rank's trials must be in the study before reading the winner
    if r != 0:
        return {}, None
    return study.best_params, study


def _save_checkpoint(path: str, payload: dict[str, Any]) -> None:
    """Atomically write the checkpoint on rank 0 (a preemption mid-write must
    never corrupt the previous good checkpoint)."""
    from .utils import is_main

    if not is_main():
        return
    import os

    import torch

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# Bump when the checkpoint payload layout changes; _load_checkpoint rejects
# unknown versions instead of silently misreading them.
_CHECKPOINT_FORMAT = 1


def _load_checkpoint(path: str | None) -> dict[str, Any] | None:
    import os

    if path is None or not os.path.exists(path):
        return None
    import torch

    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    found = ckpt.get("format_version")
    if found != _CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint {path} has format_version={found!r}, but this "
            f"autotrainer expects {_CHECKPOINT_FORMAT}. It was written by an "
            "incompatible version - delete the file to start fresh, or load "
            "it with the autotrainer version that wrote it."
        )
    return ckpt


def fit(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    *,
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
        trials: number of Optuna trials in phase 1.
        epochs: maximum full-training epochs in phase 2.
        epochs_per_trial: epochs trained per trial in phase 1.
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

    Returns:
        ``(model, best_params, study)``. The model is the plain module
        (never DDP-wrapped) carrying the best epoch's weights. Under DDP,
        ``study`` is ``None`` on every rank except rank 0.
    """
    import torch

    from .auto_optim import _infer_loss, _make_loss, _make_optimizer
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
                storage_path=study_storage or f".autotrainer_study_{key}.log",
            )
            # Ranks > 0 hold an empty dict; rank 0 read the winner.
            [best_params] = _sync_from_rank0([best_params], True)
    loss_fn = _make_loss(loss)

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
    opt, opt_name, _ = _make_optimizer(
        m,
        best_params.get("optimizer"),
        best_params.get("lr", 1e-3),
        best_params.get("weight_decay", 0.0),
    )

    steps = max(len(tl) * epochs, 1)
    warmup = max(int(0.05 * steps), 1)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps - warmup),
        ],
        milestones=[warmup],
    )
    if verbose:
        print0(
            f"[autotrainer] fit: retraining winner from original init "
            f"(optimizer={opt_name}, up to {epochs} epochs, patience={patience})"
        )

    scaler = GradScaler()
    best_val, best_state, bad_epochs, start_epoch = float("inf"), None, 0, 0
    if ckpt is not None:
        _unwrap(m).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
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
            xb_dev = to_device(xb, device)
            yb_dev = to_device(yb, device)
            opt.zero_grad()
            with autocast_context():
                out = robust_forward(m, xb_dev)
                loss_val = loss_fn(out, yb_dev)
            scaler.scale(loss_val).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        val = _evaluate(m, val_loader, loss_fn, device)
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
                    "scheduler": sched.state_dict(),
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
    return final, best_params, study
