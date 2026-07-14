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


def prepare(model: Any, dataloader: Any = None, optimizer: Any = None) -> Any:
    """Make (model, dataloader, optimizer) distribution-ready.

    Single device: returns inputs unchanged (moved to GPU if available).
    Multi device:  init process group, DDP-wrap model, distributed sampler.
    """
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    rank, local_rank, world_size = _dist_info()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    model = model.to(device)

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if use_cuda else "gloo",
                rank=rank,
                world_size=world_size,
            )
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)

        if dataloader is not None:
            sampler = DistributedSampler(dataloader.dataset, num_replicas=world_size, rank=rank)
            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=sampler,
                num_workers=dataloader.num_workers,
                pin_memory=use_cuda,
                drop_last=dataloader.drop_last,
                collate_fn=dataloader.collate_fn,
            )

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
