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


def prepare(
    model: Any,
    dataloader: Any = None,
    optimizer: Any = None,
    *,
    optimize: bool = False,
    amp: bool | None = None,
    auto_bs: bool = False,
    loss_fn: Any = None,
    max_bs: int = 4096,
) -> Any:
    """Make (model, dataloader, optimizer) distribution-ready.

    Single device: returns inputs unchanged (moved to GPU if available).
    Multi device:  init process group, DDP-wrap model, distributed sampler.

    Args:
        model: a ``torch.nn.Module``.
        dataloader: optional ``DataLoader``.
        optimizer: optional optimizer; passed through untouched.
        optimize: turn on the GPU wins users forget (TF32, cudnn.benchmark
            for CNNs, ``num_workers``/``pin_memory``/``persistent_workers``
            defaults on bare loaders). Never touches lr, loss, schedule, or
            optimizer choice. No-op on CPU.
        amp: enable mixed precision (bf16-preferred, fp16 fallback). When
            ``optimize=True`` this defaults to True; otherwise False. The
            caller uses ``autotrainer.autocast_context()`` around the
            forward and ``autotrainer.GradScaler()`` for the backward; both
            already exist as no-ops on CPU / when bf16 is supported.
        auto_bs: grow the loader's batch size until OOM, then back off one
            step. Requires ``loss_fn`` for an accurate forward+backward
            measurement; without it the sweep is forward-only (conservative
            - it underestimates, since real training also needs memory for
            grads + optimizer state). The discovered batch size replaces
            the loader's; lr and schedule are NOT changed - raise your
            effective batch with :func:`autotrainer.accumulate` instead.
        loss_fn: the loss for the ``auto_bs`` forward+backward sweep. Not
            used for anything else; nothing is inferred or overridden.
        max_bs: ceiling for the ``auto_bs`` sweep (default 4096).
    """
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    from .._optimize import apply_gpu_flags, build_loader_defaults, summarize

    rank, local_rank, world_size = _dist_info()
    # cuda_device() gates on device_count() > 0 so a driver-present but
    # GPU-hidden box doesn't try to bind a phantom device.
    from ..utils import cuda_device

    device = cuda_device(local_rank)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.set_device(device)
    model = model.to(device)

    # Optimize half: set the free-win flags before any work, so even the
    # DDP-wrap below benefits from TF32 etc. Track what we actually changed
    # so the summary is honest (apply_gpu_flags is a black box otherwise).
    applied: dict[str, Any] = {}
    if optimize and use_cuda:
        from .._optimize import _looks_like_cnn

        cnn = _looks_like_cnn(model)
        before = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.benchmark,
        )
        apply_gpu_flags(model, cnn=cnn)
        if torch.backends.cuda.matmul.allow_tf32 != before[0]:
            applied["tf32"] = True
        if cnn and torch.backends.cudnn.benchmark != before[1]:
            applied["cudnn_benchmark"] = True

    if amp is None:
        amp = optimize  # default: optimize= implies AMP, but stays overridable

    if world_size > 1:
        if dataloader is not None:
            # Shard (and validate) BEFORE any collective op: if this rank
            # raised after init, the others would hang in the process group.
            dataloader = _shard_loader(dataloader, rank, world_size)
        _ensure_process_group()
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    # Loader optimizations run AFTER sharding so they see the final loader.
    # Only adds keys the user didn't set; merges with whatever _shard_loader
    # already preserved.
    if optimize and dataloader is not None and use_cuda:
        extra = build_loader_defaults(dataloader, world_size)
        if extra:
            from torch.utils.data import DataLoader

            merged = _loader_kwargs(dataloader)
            merged.update(extra)
            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=dataloader.sampler,
                **merged,
            )
            applied.update(extra)

    # Auto batch size: sweep after sharding + loader-kwargs so the test
    # runs against the same loader the user will train with. The sweep
    # rebuilds the loader once with the discovered size.
    if auto_bs and dataloader is not None and use_cuda:
        from ..utils import print0, robust_forward, split_xy, to_device

        def sample_batch_fn(bs: int) -> None:
            from torch.utils.data import DataLoader as _DL

            tmp = _DL(dataloader.dataset, batch_size=bs, **_loader_kwargs(dataloader))
            xb, yb = split_xy(next(iter(tmp)))
            xb_dev = to_device(xb, device)
            opt_zero = torch.optim.SGD(model.parameters(), lr=1e-6)  # throwaway
            opt_zero.zero_grad(set_to_none=True)
            out = robust_forward(model, xb_dev)
            if loss_fn is not None:
                yb_dev = to_device(yb, device)
                loss_fn(out, yb_dev).backward()
                opt_zero.step()
            else:
                # Forward-only sweep: sum outputs to keep the graph, backprop
                # a scalar. Underestimates the safe batch (no real grads +
                # optimizer state) but never overestimates it.
                out.sum().backward()
                opt_zero.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        old_bs = dataloader.batch_size
        new_bs = find_batch_size(model, sample_batch_fn, start=max(2, old_bs), max_bs=max_bs)
        if new_bs != old_bs:
            from torch.utils.data import DataLoader

            merged = _loader_kwargs(dataloader)
            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=new_bs,
                sampler=dataloader.sampler,
                **merged,
            )
            applied["batch_size"] = (old_bs, new_bs)
            print0(
                f"[autotrainer] optimize: batch_size {old_bs} -> {new_bs} "
                f"(lr and schedule unchanged; use accumulate() to scale the step)"
            )

    if optimize:
        summarize(optimize=optimize, amp=bool(amp and use_cuda), applied=applied)

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
