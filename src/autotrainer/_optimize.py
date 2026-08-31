"""GPU optimization flags applied by ``prepare(..., optimize=True)``.

The original idea of autotrainer: detect the hardware, set it up correctly,
and **leave the user's hyperparameters alone**. This module holds the
"optimize" half of that contract - the things that are almost always a
free win on modern CUDA hardware and that users forget (or don't know) to
set. None of these change lr / loss / schedule / optimizer choice.

Everything here is a no-op on CPU or when the flag is off, so the existing
``prepare()`` behavior is unchanged for callers that don't opt in.
"""

from __future__ import annotations

from typing import Any

# Loader keys _loader_kwargs already carries over. These are the ones it
# leaves at the torch default because the user didn't set them - the ones
# where a sane autodetected default beats torch's default.
_DEFAULT_NUM_WORKERS_CAP = 8


def _looks_like_cnn(model: Any) -> bool:
    import torch.nn as nn

    return any(isinstance(m, (nn.Conv2d, nn.Conv3d)) for m in model.modules())


def _is_conv2d_net(model: Any) -> bool:
    """True for a net whose convolutions are *all* ``Conv2d``.

    ``channels_last`` is a 4D layout. A Conv1d net would get a silent no-op,
    and a Conv3d net wants ``channels_last_3d`` instead - converting it to the
    4D format is not what the caller means. Requiring every conv to be 2D
    keeps the conversion to the case where it is unambiguously right.
    """
    import torch.nn as nn

    convs = [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))]
    return bool(convs) and all(isinstance(m, nn.Conv2d) for m in convs)


def apply_channels_last(model: Any) -> bool:
    """Convert a conv net to NHWC, which is what the tensor cores want.

    ONLY call this when AMP is on. The layout is a large win under mixed
    precision (~+28% on a 4-conv net, RTX 5070) and a large LOSS without it:
    measured 3x SLOWER in fp32/TF32, because the fp32 kernels are the ones
    cuDNN has NCHW fast paths for and the conversion just adds transposes.
    A "free win" that triples fp32 step time is not a free win, so the caller
    gates on ``amp`` and this stays a no-op everywhere else.

    Returns whether the conversion was applied.
    """
    import torch

    if not _is_conv2d_net(model):
        return False
    model.to(memory_format=torch.channels_last)
    return True


def apply_gpu_flags(model: Any, *, cnn: bool | None = None) -> None:
    """Toggle the global PyTorch backend flags that are almost always a win.

    Idempotent and safe to call from any rank.

    * ``cudnn.benchmark = True`` when conv layers are present (fixed input
      shape assumed - the common case). torch's own docs recommend this.
    * ``cuda.matmul.allow_tf32 = True`` and ``cudnn.allow_tf32 = True`` on
      Ampere+: a ~2-3x matmul win for free. Shipped disabled by default for
      legacy reproducibility reasons that almost never apply.
    """
    import torch

    if not torch.cuda.is_available():
        return

    if cnn is None:
        cnn = _looks_like_cnn(model)
    if cnn:
        torch.backends.cudnn.benchmark = True

    # TF32: all Ampere+ GPUs (A100, H100, RTX 30/40, L40, A10, ...). This is
    # the single highest payoff-per-line knob in modern PyTorch.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        # Older torch: the flag didn't exist yet. Silent no-op.
        pass


def build_loader_defaults(dataloader: Any, world_size: int) -> dict[str, Any]:
    """Return kwargs that improve a bare ``DataLoader`` without clobbering
    user intent.

    Only acts on values the user *didn't* set (i.e. torch defaults). A user
    who wrote ``DataLoader(ds, batch_size=32)`` gets ``num_workers`` etc.
    added; a user who wrote ``DataLoader(ds, num_workers=4, ...)`` is left
    alone. Returns an empty dict on CPU.
    """
    import torch

    if not torch.cuda.is_available():
        return {}

    kwargs: dict[str, Any] = {}

    # num_workers=0 is torch's default and the #1 silent cause of GPU
    # starvation. One worker per CPU this rank may actually use - which under
    # SLURM is the allocation, not the node's core count. Sizing from the node
    # spawns 8 workers into a 2-CPU allocation and they thrash.
    if dataloader.num_workers == 0 and not dataloader.persistent_workers:
        from .utils import available_cpus

        kwargs["num_workers"] = min(available_cpus(world_size), _DEFAULT_NUM_WORKERS_CAP)

    # pin_memory defaults to False; free H2D overlap on CUDA.
    if not dataloader.pin_memory:
        kwargs["pin_memory"] = True

    # persistent_workers avoids respawning workers every epoch. Only valid
    # when num_workers > 0.
    final_nw = kwargs.get("num_workers", dataloader.num_workers)
    if final_nw > 0 and not dataloader.persistent_workers:
        kwargs["persistent_workers"] = True

    return kwargs


def summarize(
    optimize: bool,
    amp: bool,
    applied: dict[str, Any],
    *,
    compile: bool = False,
    fsdp: bool = False,
) -> None:
    """Print what was applied, so the user can see it - a silent optimization
    that changes wall-clock 2x is worse than no optimization at all.

    Runs when ``optimize=True`` OR when any explicit opt-in (``compile``,
    ``fsdp``) fired - those are user requests that deserve a confirmation
    line even without the broader optimize bundle.
    """
    from .utils import print0

    if not optimize and not compile and not fsdp:
        return
    parts = []
    if applied.get("tf32"):
        parts.append("TF32")
    if applied.get("cudnn_benchmark"):
        parts.append("cudnn.benchmark")
    if applied.get("channels_last"):
        parts.append("channels_last")
    if applied.get("num_workers") is not None:
        parts.append(f"num_workers={applied['num_workers']}")
    if applied.get("pin_memory"):
        parts.append("pin_memory")
    if applied.get("persistent_workers"):
        parts.append("persistent_workers")
    if amp:
        parts.append("AMP")
    if applied.get("compile"):
        parts.append(f"torch.compile(mode={applied['compile']})")
    if applied.get("wrap") == "fsdp":
        parts.append("FSDP")
    if applied.get("cpu_offload"):
        parts.append("CPU-offload")
    if applied.get("ddp_opts"):
        parts.append(f"DDP[{'+'.join(applied['ddp_opts'])}]")
    if parts:
        print0(f"[autotrainer] optimize: {', '.join(parts)} (hyperparameters untouched)")
