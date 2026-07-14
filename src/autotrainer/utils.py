"""Rank-aware utilities and mixed precision.

In distributed training, every process runs the same script - so without
guards you get N copies of every print and N processes fighting to write
the same checkpoint file. These helpers make "only rank 0 does it" a
one-liner.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


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


def get_model_device(model) -> torch.device:
    """Get the device of the model's first parameter, defaulting to CPU."""
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def to_device(data, device):
    """Recursively move tensors in dictionaries, lists, tuples, or raw tensors to device."""
    import torch

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    return data


def slice_batch(data, n: int = 2):
    """Recursively slice batch data structures to the first n samples."""
    import torch

    if isinstance(data, torch.Tensor):
        return data[:n]
    elif isinstance(data, dict):
        return {k: slice_batch(v, n) for k, v in data.items()}
    elif isinstance(data, list):
        return [slice_batch(v, n) for v in data]
    elif isinstance(data, tuple):
        return tuple(slice_batch(v, n) for v in data)
    return data


def robust_forward(model, xb):
    """Call a model robustly with dict unpacking, list/tuple unpacking, or direct args."""
    if isinstance(xb, dict):
        try:
            return model(**xb)
        except TypeError:
            return model(xb)
    elif isinstance(xb, (list, tuple)):
        try:
            return model(*xb)
        except TypeError:
            return model(xb)
    return model(xb)


def get_batch_size(data) -> int:
    """Recursively find the batch size of the target or input data structure."""
    import torch

    if isinstance(data, torch.Tensor):
        return data.shape[0]
    elif isinstance(data, dict):
        for v in data.values():
            sz = get_batch_size(v)
            if sz > 0:
                return sz
    elif isinstance(data, (list, tuple)):
        for v in data:
            sz = get_batch_size(v)
            if sz > 0:
                return sz
    return 0


def GradScaler(*args, **kwargs):
    """Return a GradScaler enabled only if fp16 is active.

    If CUDA is not available or if the GPU supports bf16, this returns a disabled
    GradScaler (which behaves as a no-op / pass-through).
    """
    import torch

    enabled = False
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        enabled = True
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", *args, enabled=enabled, **kwargs)
    return torch.cuda.amp.GradScaler(*args, enabled=enabled, **kwargs)
