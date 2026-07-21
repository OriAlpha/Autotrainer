"""PyTorch backend.

Called from user code via `autotrainer.prepare(...)`. Reads the env vars the
launcher set, initializes the process group if world_size > 1, wraps the
model in DDP, and swaps the DataLoader's sampler for a DistributedSampler.
"""

from __future__ import annotations

import os
from typing import Any


def _dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _ensure_process_group() -> bool:
    """Init the process group once if WORLD_SIZE > 1; True when distributed."""
    import torch
    import torch.distributed as dist

    rank, local_rank, world_size = _dist_info()
    if world_size <= 1:
        return False
    if not dist.is_initialized():
        # cuda_device() gates on device_count() > 0, not just is_available(),
        # so a driver-present but GPU-hidden box (e.g. CUDA_VISIBLE_DEVICES="")
        # falls through to gloo instead of crashing in set_device(N).
        from ..utils import cuda_device

        use_cuda = cuda_device(local_rank).type == "cuda"
        if use_cuda:
            # Bind before NCCL init so ranks don't all land on GPU 0.
            torch.cuda.set_device(local_rank)
        kwargs: dict[str, Any] = {}
        timeout_s = os.environ.get("AUTOTRAINER_TIMEOUT")
        if timeout_s:
            # Long rank-0 phases (e.g. fit()'s tuning) can outlive the
            # default collective timeout while the other ranks wait.
            from datetime import timedelta

            kwargs["timeout"] = timedelta(seconds=int(timeout_s))
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo",
            rank=rank,
            world_size=world_size,
            **kwargs,
        )
    return True


def _loader_kwargs(dataloader: Any) -> dict[str, Any]:
    """Carry the user's DataLoader settings over to a rebuilt loader."""
    kwargs: dict[str, Any] = {
        "num_workers": dataloader.num_workers,
        "collate_fn": dataloader.collate_fn,
        "pin_memory": dataloader.pin_memory,
        "drop_last": dataloader.drop_last,
        "timeout": dataloader.timeout,
        "worker_init_fn": dataloader.worker_init_fn,
        "generator": dataloader.generator,
        "persistent_workers": dataloader.persistent_workers,
    }
    if dataloader.num_workers > 0:
        # DataLoader rejects prefetch_factor when there are no workers.
        kwargs["prefetch_factor"] = dataloader.prefetch_factor
    return kwargs


def _shard_loader(dataloader: Any, rank: int, world_size: int) -> Any:
    """Swap the loader's sampler for a DistributedSampler, keeping its settings."""
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    from torch.utils.data.distributed import DistributedSampler

    if isinstance(dataloader.dataset, torch.utils.data.IterableDataset):
        raise TypeError(
            "autotrainer cannot shard a DataLoader over an IterableDataset; "
            "shard inside the dataset (e.g. by RANK/WORLD_SIZE env vars) and "
            "it will be passed through unchanged."
        )
    if isinstance(dataloader.sampler, DistributedSampler):
        return dataloader  # the user already sharded it
    if dataloader.batch_size is None:
        raise TypeError(
            "autotrainer cannot shard a DataLoader built with batch_sampler=; "
            "construct it with batch_size= and let autotrainer install the "
            "DistributedSampler, or pass sampler=DistributedSampler(...) yourself."
        )
    # Honor the user's shuffle choice: SequentialSampler means shuffle=False.
    shuffle = not isinstance(dataloader.sampler, SequentialSampler)
    sampler = DistributedSampler(
        dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )
    if rank == 0:
        print(
            "[autotrainer] DistributedSampler installed (shuffle="
            f"{shuffle}) - call autotrainer.set_epoch(loader, epoch) at the "
            "start of every epoch so each epoch reshuffles"
        )
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        **_loader_kwargs(dataloader),
    )


def prepare(model: Any, dataloader: Any = None, optimizer: Any = None) -> Any:
    """Make (model, dataloader, optimizer) distribution-ready.

    Single device: returns inputs unchanged (moved to GPU if available).
    Multi device:  init process group, DDP-wrap model, distributed sampler.
    """
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank, local_rank, world_size = _dist_info()
    # cuda_device() gates on device_count() > 0 so a driver-present but
    # GPU-hidden box doesn't try to bind a phantom device.
    from ..utils import cuda_device

    device = cuda_device(local_rank)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.set_device(device)
    model = model.to(device)

    if world_size > 1:
        if dataloader is not None:
            # Shard (and validate) BEFORE any collective op: if this rank
            # raised after init, the others would hang in the process group.
            dataloader = _shard_loader(dataloader, rank, world_size)
        _ensure_process_group()
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    out = [model]
    if dataloader is not None:
        out.append(dataloader)
    if optimizer is not None:
        out.append(optimizer)
    return out[0] if len(out) == 1 else tuple(out)


def find_batch_size(model: Any, sample_batch_fn: Any, start: int = 2, max_bs: int = 4096) -> int:
    """Double batch size until OOM, then back off one step.

    sample_batch_fn(bs) must run one forward+backward pass at batch size bs.
    """
    import torch

    bs = start
    best = start
    while bs <= max_bs:
        try:
            sample_batch_fn(bs)
            best = bs
            bs *= 2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            raise
    return best
