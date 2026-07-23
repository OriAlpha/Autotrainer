"""Phase-1 (search) and checkpoint helpers, split out of ``fitting.py``.

``fit()`` composes two phases - an Optuna recipe search (phase 1) and a full
retrain of the winner (phase 2). This module holds the phase-1 machinery and
the preemption-safe checkpoint read/write, so ``fitting.py`` can stay focused
on the orchestration and the phase-2 training loop. Pure extraction - no
behavior change; the names are re-exported from ``fitting.py`` so existing
``from autotrainer.fitting import _unwrap, _load_checkpoint, ...`` keeps
working.

Under ``autotrainer run`` with multiple processes, the search itself is
parallel: trials are split across the ranks and pulled from a shared Optuna
journal-file study, one trial per process on its own device.
"""

from __future__ import annotations

from typing import Any

from .tuning import tune


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
