"""Auto model optimization (PyTorch).

Smart defaults, not AutoML: infer sensible one-shot choices from the model
and data instead of searching. Design rules:

    1. Every decision is PRINTED with its reasoning - a silently wrong loss
       function trains fine and produces garbage.
    2. Every decision is OVERRIDABLE via keyword arguments.

    model, loader, opt, loss_fn = autotrainer.auto(model, loader)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

LOSSES = ("cross_entropy", "bce", "mse", "huber")


def _peek_batch(dataloader: Any) -> tuple[Any, Any]:
    xb, yb = next(iter(dataloader))
    return xb, yb


def _bce_loss() -> nn.Module:
    """BCEWithLogitsLoss that also accepts integer class targets shaped (N,).

    Plain BCEWithLogitsLoss requires FLOAT targets shaped like the (N, 1)
    logits, but binary-classification datasets almost always yield integer
    (N,) labels - the exact shape _infer_loss saw when it picked BCE. Without
    this adapter the inferred loss would raise on the user's very next batch.
    """
    import torch
    import torch.nn as nn

    class _BCEWithLogitsAdapter(nn.BCEWithLogitsLoss):  # type: ignore[misc]
        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if not torch.is_floating_point(target):
                target = target.float()
            if target.shape != input.shape:
                target = target.reshape(input.shape)
            return super().forward(input, target)

    return _BCEWithLogitsAdapter()


def _infer_loss(model: Any, yb: Any, xb: Any) -> tuple[nn.Module, str, str]:
    """Pick a loss from target dtype/shape, sanity-checked against model output."""
    import torch
    import torch.nn as nn

    from .utils import get_model_device, robust_forward, slice_batch, to_device

    device = get_model_device(model)
    xb_dev = to_device(xb, device)
    yb = to_device(yb, device)

    with torch.no_grad():
        xb_slice = slice_batch(xb_dev, 2)
        out = robust_forward(model, xb_slice)
    out_dim = out.shape[-1] if out.ndim > 1 else 1

    if not torch.is_floating_point(yb):
        n_classes = int(yb.max().item()) + 1
        # Binary classification (exactly 2 classes) with a single output -> BCE.
        # Parentheses required: without them `and` binds tighter than `or` and
        # the n_classes check becomes dead code (any 1-output model -> BCE).
        if n_classes == 2 and out_dim == 1:
            reason = f"integer targets, binary ({n_classes} classes), 1 output"
            return _bce_loss(), "bce", reason
        reason = f"integer targets with {n_classes} classes, model outputs {out_dim}"
        if out_dim < n_classes:
            reason += f" [WARNING: output dim {out_dim} < {n_classes} classes!]"
        return nn.CrossEntropyLoss(), "cross_entropy", reason

    # float targets -> regression; Huber if outlier-heavy.
    # Use median/MAD, not mean/std: outliers inflate the std and hide themselves.
    y = yb.float().flatten()
    med = y.median()
    mad = (y - med).abs().median() * 1.4826  # ~std for normal data
    if mad > 0:  # noqa: SIM108
        outliers = ((y - med).abs() > 3 * mad).float().mean().item()
    else:
        outliers = 0.0
    if outliers > 0.01:
        return nn.HuberLoss(), "huber", f"float targets, {outliers:.1%} outliers beyond 3 MAD"
    return nn.MSELoss(), "mse", "float targets, no heavy outliers"


def _make_loss(name: str) -> nn.Module:
    import torch.nn as nn

    if name == "bce":
        return _bce_loss()
    return {
        "cross_entropy": nn.CrossEntropyLoss,
        "mse": nn.MSELoss,
        "huber": nn.HuberLoss,
    }[name]()


def _looks_like_cnn(model: Any) -> bool:
    import torch.nn as nn

    return any(isinstance(m, (nn.Conv2d, nn.Conv3d)) for m in model.modules())


def _param_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """Exclude biases and norm params from weight decay (the common mistake)."""
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _make_optimizer(
    model: Any, name: str | None, lr: float, weight_decay: float
) -> tuple[Any, str, str]:
    import torch

    groups = _param_groups(model, weight_decay)
    if name is None:
        name = "sgd" if _looks_like_cnn(model) else "adamw"
        reason = (
            "conv layers detected -> SGD+momentum (classic CNN recipe)"
            if name == "sgd"
            else "general default -> AdamW"
        )
    else:
        reason = "user override"
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True), name, reason
    return torch.optim.AdamW(groups, lr=lr), name, reason


def find_lr(
    model: Any,
    dataloader: Any,
    loss_fn: Any,
    optimizer_name: str = "adamw",
    min_lr: float = 1e-7,
    max_lr: float = 1.0,
    num_iters: int = 100,
) -> float:
    """LR range test (Leslie Smith): sweep LR exponentially, track loss,
    return the LR at the steepest descent.

    Runs on a deep-copied throwaway model so the real weights are untouched.
    The sweep stops early if the loss diverges (non-finite or 4x the best
    seen). The returned LR is the steepest-descent point backed off 10x for
    stability; if the sweep fails (too few samples), a safe ``3e-4`` default
    is returned.

    Note: this runs locally in the calling process. Under DDP, ``auto()``
    wraps it so the sweep happens on rank 0 only and the result is broadcast
    - if you call ``find_lr`` directly in a distributed script, do the same
    (differently-shuffled loaders can otherwise land each rank on a
    different LR).

    Args:
        model: a ``torch.nn.Module``.
        dataloader: batches to train on during the sweep.
        loss_fn: the loss to minimize.
        optimizer_name: ``"adamw"`` or ``"sgd"`` for the sweep.
        min_lr: starting LR of the exponential sweep.
        max_lr: ending LR of the exponential sweep.
        num_iters: number of LR steps to evaluate.

    Returns:
        The suggested learning rate.
    """
    import copy
    import math
    import os

    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    from .utils import robust_forward, to_device

    m = copy.deepcopy(model).to(device)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    m.train()
    opt, _, _ = _make_optimizer(m, optimizer_name, lr=min_lr, weight_decay=0.0)
    gamma = (max_lr / min_lr) ** (1 / max(num_iters - 1, 1))

    lrs, losses, lr, it = [], [], min_lr, 0
    smoothed, best = None, float("inf")
    try:
        while it < num_iters:
            for xb, yb in dataloader:
                if it >= num_iters:
                    break
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad()
                xb_dev = to_device(xb, device)
                yb_dev = to_device(yb, device)
                out = robust_forward(m, xb_dev)
                loss = loss_fn(out, yb_dev)
                loss.backward()
                opt.step()

                v = loss.item()
                smoothed = v if smoothed is None else 0.9 * smoothed + 0.1 * v
                if not math.isfinite(smoothed) or smoothed > 4 * best:
                    it = num_iters  # diverged - stop early
                    break
                best = min(best, smoothed)
                lrs.append(lr)
                losses.append(smoothed)
                lr *= gamma
                it += 1
    finally:
        del m, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(losses) < 5:
        return 3e-4  # sweep failed; safe default

    # steepest descent point, backed off by 10x for stability
    grads = [(losses[i + 1] - losses[i]) for i in range(len(losses) - 1)]
    steepest = min(range(len(grads)), key=lambda i: grads[i])
    return max(lrs[steepest] / 10, min_lr)


def _find_lr_synced(model: Any, dataloader: Any, loss_fn: Any, optimizer_name: str) -> float:
    """find_lr on rank 0 only; the other ranks receive the result by broadcast.

    Under DDP every rank runs auto(), and letting each rank sweep
    independently wastes work and - worse - differently-shuffled loaders can
    land each rank on a different LR. Gradients are synced but every
    optimizer steps with its own local LR, so the replicas silently drift.
    """
    from .backends.torch_backend import _ensure_process_group

    if not _ensure_process_group():
        return find_lr(model, dataloader, loss_fn, optimizer_name)

    import os

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    lr = find_lr(model, dataloader, loss_fn, optimizer_name) if rank == 0 else 0.0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # NCCL broadcasts need a CUDA tensor; gloo needs CPU.
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([lr], dtype=torch.float64, device=device)
    dist.broadcast(t, src=0)
    return float(t.item())


def auto(
    model: Any,
    dataloader: Any,
    *,
    loss: str | None = None,
    optimizer: str | None = None,
    lr: float | None = None,
    weight_decay: float = 0.01,
    schedule: bool = True,
    epochs: int = 10,
) -> tuple[Any, ...]:
    """Infer loss/optimizer/LR/schedule, then distribute the model.

    Smart defaults, not AutoML: every inferred choice is printed with its
    reasoning and can be overridden via the keyword arguments below. Runs
    BEFORE distribution - it infers on the raw model, then calls
    ``prepare()`` for DDP/device placement.

    Args:
        model: a ``torch.nn.Module``.
        dataloader: a training DataLoader; one batch is peeked to infer the
            loss from the target dtype/shape.
        loss: override the inferred loss; one of ``"cross_entropy"``,
            ``"bce"``, ``"mse"``, ``"huber"``. If ``None``, inferred.
        optimizer: ``"adamw"`` or ``"sgd"``; if ``None``, picked from the
            architecture (SGD for CNNs, AdamW otherwise).
        lr: learning rate; if ``None``, found via an LR range test.
        weight_decay: decoupled weight decay (excluded from biases/norms).
        schedule: if True, build a warmup(5%)+cosine schedule.
        epochs: assumed epoch count, used to size the schedule.

    Returns:
        ``(model, dataloader, optimizer, loss_fn, scheduler)``. The
        scheduler is ``None`` when ``schedule=False``, so the shape of the
        return value never changes.
    """
    from .backends.torch_backend import prepare

    xb, yb = _peek_batch(dataloader)

    if loss is not None:
        loss_fn, loss_name, loss_why = _make_loss(loss), loss, "user override"
    else:
        loss_fn, loss_name, loss_why = _infer_loss(model, yb, xb)

    if lr is not None:
        lr_val, lr_why = lr, "user override"
    else:
        lr_val = _find_lr_synced(model, dataloader, loss_fn, optimizer or "adamw")
        lr_why = "LR range test (steepest descent / 10, rank 0)"

    opt, opt_name, opt_why = _make_optimizer(model, optimizer, lr_val, weight_decay)

    print(f"[autotrainer] auto: loss={loss_name} ({loss_why})")
    print(
        f"[autotrainer] auto: optimizer={opt_name} ({opt_why}), "
        f"weight_decay={weight_decay} (excluded from biases/norms)"
    )
    print(f"[autotrainer] auto: lr={lr_val:.2e} ({lr_why})")

    model, dataloader, opt = prepare(model, dataloader, opt)

    sched = None
    if schedule:
        import torch

        steps = max(len(dataloader) * epochs, 1)
        warmup = max(int(0.05 * steps), 1)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [
                torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps - warmup),
            ],
            milestones=[warmup],
        )
        print(
            f"[autotrainer] auto: schedule=warmup({warmup} steps)+cosine "
            f"(assumes {epochs} epochs; pass epochs=N to change)"
        )

    return model, dataloader, opt, loss_fn, sched
