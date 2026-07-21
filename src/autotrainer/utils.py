"""Rank-aware utilities and mixed precision.

In distributed training, every process runs the same script - so without
guards you get N copies of every print and N processes fighting to write
the same checkpoint file. These helpers make "only rank 0 does it" a
one-liner.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def cuda_device(local_rank: int = 0) -> torch.device:
    """The CUDA device for ``local_rank``, or CPU when no GPU is visible.

    ``torch.cuda.is_available()`` is True whenever the driver is present,
    even when ``CUDA_VISIBLE_DEVICES=""`` hides every device - so callers
    that built ``cuda:{local_rank}`` from it alone would call
    ``set_device(N)`` on a phantom GPU and crash ("invalid device
    ordinal"). ``device_count() > 0`` is the real signal; this helper
    centralizes the check so every device-pick site stays consistent.
    """
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def is_main() -> bool:
    """True on exactly one process (global rank 0)."""
    return rank() == 0


def print0(*args: Any, **kwargs: Any) -> None:
    """Print only on rank 0 - use for logging inside training loops."""
    if is_main():
        print(*args, **kwargs)


def save0(obj: Any, path: str) -> None:
    """torch.save, but only on rank 0, so workers don't clobber the file."""
    if is_main():
        import torch

        torch.save(obj, path)


def set_epoch(dataloader: Any, epoch: int) -> None:
    """Tell a DistributedSampler which epoch this is - call at every epoch start.

    Without it, a DistributedSampler yields the SAME shuffle order every
    epoch. No-op for non-distributed loaders, so it is always safe:

        for epoch in range(epochs):
            autotrainer.set_epoch(loader, epoch)
            for xb, yb in loader:
                ...
    """
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


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


def autocast_context() -> Any:
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


def get_model_device(model: Any) -> torch.device:
    """Get the device of the model's first parameter, defaulting to CPU."""
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def to_device(data: Any, device: Any) -> Any:
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


def slice_batch(data: Any, n: int = 2) -> Any:
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


def split_xy(batch: Any) -> tuple[Any, Any]:
    """Split one dataloader batch into ``(inputs, targets)``.

    Handles the batch shapes real datasets yield, not just the ``(xb, yb)``
    2-tuple the examples use - so ``auto``/``fit``/``tune`` don't die on an
    opaque unpack error the first time someone brings their own loader:

      * ``(xb, yb)`` or ``(xb, yb, ...)`` - first is input, second is target;
        extra elements (e.g. sample weights) are ignored for inference.
      * ``(xb,)`` or a bare tensor - input only; target is ``None``.
      * a ``dict`` (HuggingFace-style) - the value under the first matching
        label key (``labels``/``label``/``targets``/``target``/``y``) is the
        target and the remaining keys are the input; no label key -> target
        ``None`` and the whole dict is the input.

    A ``None`` target means the loss can't be inferred - callers should ask the
    user to pass ``loss=`` explicitly rather than fail cryptically downstream.
    """
    if isinstance(batch, dict):
        for key in ("labels", "label", "targets", "target", "y"):
            if key in batch:
                xb = {k: v for k, v in batch.items() if k != key}
                return (xb or batch), batch[key]
        return batch, None
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        if len(batch) == 1:
            return batch[0], None
        raise ValueError("dataloader yielded an empty batch")
    # A bare tensor / array: inputs only, no targets to infer a loss from.
    return batch, None


def robust_forward(model: Any, xb: Any) -> Any:
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


def get_batch_size(data: Any) -> int:
    """Recursively find the batch size of the target or input data structure."""
    import torch

    if isinstance(data, torch.Tensor):
        return int(data.shape[0])
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


def GradScaler(*args: Any, **kwargs: Any) -> Any:
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
