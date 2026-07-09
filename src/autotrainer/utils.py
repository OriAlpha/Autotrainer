"""Rank-aware utilities and mixed precision.

In distributed training, every process runs the same script - so without
guards you get N copies of every print and N processes fighting to write
the same checkpoint file. These helpers make "only rank 0 does it" a
one-liner.
"""

from __future__ import annotations

import os


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main() -> bool:
    """True on exactly one process (global rank 0)."""
    return rank() == 0


def print0(*args, **kwargs) -> None:
    """Print only on rank 0 - use for logging inside training loops."""
    if is_main():
        print(*args, **kwargs)


def save0(obj, path: str) -> None:
    """torch.save, but only on rank 0, so workers don't clobber the file."""
    if is_main():
        import torch
        torch.save(obj, path)


def barrier() -> None:
    """Wait for all workers (no-op when not distributed).

    Typical use: barrier() after rank 0 downloads a dataset, so other
    ranks don't start reading a half-written file.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except ImportError:
        pass


def autocast_context():
    """Mixed-precision context: bf16 on modern GPUs, fp16 otherwise, no-op on CPU.

    Usage:
        with autotrainer.autocast_context():
            loss = loss_fn(model(x), y)
    """
    import contextlib

    import torch

    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
