"""Auto model optimization (PyTorch).

Smart defaults, not AutoML: infer sensible one-shot choices from the model
and data instead of searching. Design rules:

    1. Every decision is PRINTED with its reasoning - a silently wrong loss
       function trains fine and produces garbage.
    2. Every decision is OVERRIDABLE via keyword arguments.

    model, loader, opt, loss_fn = autotrainer.auto(model, loader)
"""

from __future__ import annotations

LOSSES = ("cross_entropy", "bce", "mse", "huber")


def _peek_batch(dataloader):
    xb, yb = next(iter(dataloader))
    return xb, yb


def _infer_loss(model, yb, xb):
    """Pick a loss from target dtype/shape, sanity-checked against model output."""
    import torch
    import torch.nn as nn
    from .utils import get_model_device, to_device, slice_batch, robust_forward

    device = get_model_device(model)
    xb_dev = to_device(xb, device)
    yb = to_device(yb, device)

    with torch.no_grad():
        xb_slice = slice_batch(xb_dev, 2)
        out = robust_forward(model, xb_slice)
    out_dim = out.shape[-1] if out.ndim > 1 else 1

    if not torch.is_floating_point(yb):
        n_classes = int(yb.max().item()) + 1
        if out_dim == 1 or n_classes == 2 and out_dim == 1:
            reason = f"integer targets, binary ({n_classes} classes), 1 output"
            return nn.BCEWithLogitsLoss(), "bce", reason
        reason = f"integer targets with {n_classes} classes, model outputs {out_dim}"
        if out_dim < n_classes:
            reason += f" [WARNING: output dim {out_dim} < {n_classes} classes!]"
        return nn.CrossEntropyLoss(), "cross_entropy", reason

    # float targets -> regression; Huber if outlier-heavy.
    # Use median/MAD, not mean/std: outliers inflate the std and hide themselves.
    y = yb.float().flatten()
    med = y.median()
    mad = (y - med).abs().median() * 1.4826  # ~std for normal data
    if mad > 0:
        outliers = ((y - med).abs() > 3 * mad).float().mean().item()
    else:
        outliers = 0.0
    if outliers > 0.01:
        return nn.HuberLoss(), "huber", f"float targets, {outliers:.1%} outliers beyond 3 MAD"
    return nn.MSELoss(), "mse", "float targets, no heavy outliers"


def _make_loss(name: str):
    import torch.nn as nn
    return {"cross_entropy": nn.CrossEntropyLoss, "bce": nn.BCEWithLogitsLoss,
            "mse": nn.MSELoss, "huber": nn.HuberLoss}[name]()


def _looks_like_cnn(model) -> bool:
    import torch.nn as nn
    return any(isinstance(m, (nn.Conv2d, nn.Conv3d)) for m in model.modules())


def _param_groups(model, weight_decay: float):
    """Exclude biases and norm params from weight decay (the common mistake)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def _make_optimizer(model, name: str | None, lr: float, weight_decay: float):
    import torch

    groups = _param_groups(model, weight_decay)
    if name is None:
        name = "sgd" if _looks_like_cnn(model) else "adamw"
        reason = ("conv layers detected -> SGD+momentum (classic CNN recipe)"
                  if name == "sgd" else "general default -> AdamW")
    else:
        reason = "user override"
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True), name, reason
    return torch.optim.AdamW(groups, lr=lr), name, reason


def find_lr(model, dataloader, loss_fn, optimizer_name: str = "adamw",
            min_lr: float = 1e-7, max_lr: float = 1.0, num_iters: int = 100) -> float:
    """LR range test (Leslie Smith): sweep LR exponentially, track loss,
    return the LR at the steepest descent. Runs on a throwaway copy of the
    model so real weights are untouched.
    """
    import copy
    import math

    import torch

    import os
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    from .utils import to_device, robust_forward

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


def auto(model, dataloader, *, loss: str | None = None, optimizer: str | None = None,
         lr: float | None = None, weight_decay: float = 0.01,
         schedule: bool = True, epochs: int = 10):
    """Returns (model, dataloader, optimizer, loss_fn[, scheduler]).

    Runs BEFORE distribution: infers loss/optimizer/lr on the raw model,
    then calls prepare() to handle DDP/device placement.
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
        lr_val = find_lr(model, dataloader, loss_fn, optimizer or "adamw")
        lr_why = "LR range test (steepest descent / 10)"

    opt, opt_name, opt_why = _make_optimizer(model, optimizer, lr_val, weight_decay)

    print(f"[autotrainer] auto: loss={loss_name} ({loss_why})")
    print(f"[autotrainer] auto: optimizer={opt_name} ({opt_why}), "
          f"weight_decay={weight_decay} (excluded from biases/norms)")
    print(f"[autotrainer] auto: lr={lr_val:.2e} ({lr_why})")

    model, dataloader, opt = prepare(model, dataloader, opt)

    if schedule:
        import torch
        steps = max(len(dataloader) * epochs, 1)
        warmup = max(int(0.05 * steps), 1)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warmup),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps - warmup)],
            milestones=[warmup],
        )
        print(f"[autotrainer] auto: schedule=warmup({warmup} steps)+cosine "
              f"(assumes {epochs} epochs; pass epochs=N to change)")
        return model, dataloader, opt, loss_fn, sched

    return model, dataloader, opt, loss_fn
