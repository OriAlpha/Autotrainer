"""Strength-scaled image augmentation - a searchable regularizer, not a policy.

Augmentation strength is one of the few knobs that genuinely moves final
quality on vision tasks, so ``tune()``/``fit()`` search it like any other
recipe entry. This module supplies the one primitive that makes that
possible: a single scalar ``strength`` that dials a fixed, well-understood
policy from "off" to "standard".

Two deliberate constraints keep it safe to search:

  * **Pure torch, no torchvision.** autotrainer's only hard dependency is
    psutil and torch is already optional; an augmentation knob must not drag
    in a new one. Everything here is tensor ops.
  * **Target-preserving.** Flip and cutout change ``x`` only, so the loss
    path is untouched. Label-mixing augmentations (mixup/cutmix) would also
    need to rewrite ``y`` and the loss, which is a larger change to the
    contract than a searchable scalar should make.

The policy is applied per-sample (not per-batch), so a batch gets a mix of
augmented and clean examples the way a per-sample transform pipeline would.
It is a no-op on anything that isn't a float NCHW image batch, which is what
lets the same call site sit in a loop that also trains tabular or text
models.
"""

from __future__ import annotations

from typing import Any

# Strength at which the policy reaches its standard setting: a 50% flip
# probability and a cutout hole of half the shorter image side. Searching
# past this point mostly destroys signal, so the default space stops here.
MAX_STRENGTH = 0.5


def augment_batch(x: Any, strength: float) -> Any:
    """Apply strength-scaled flip + cutout to an image batch.

    Args:
        x: the input batch. Augmented only when it is a floating-point 4D
            ``(N, C, H, W)`` tensor; returned unchanged otherwise, so callers
            don't need to know whether the task is vision.
        strength: ``0.0`` disables augmentation entirely (returns ``x``
            as-is). Scales both the flip probability and the cutout hole
            size up to :data:`MAX_STRENGTH`, where the policy matches the
            standard "flip + half-side cutout" setting. Values above
            :data:`MAX_STRENGTH` are clamped.

    Returns:
        A new tensor when augmentation applied, else ``x`` itself. The input
        is never mutated in place - the training batch may be a view into a
        cached dataset tensor.
    """
    if strength <= 0.0:
        return x

    import torch

    if not torch.is_tensor(x) or x.ndim != 4 or not torch.is_floating_point(x):
        return x

    s = min(float(strength), MAX_STRENGTH)
    n, _, h, w = x.shape
    out = x

    # 1. Random horizontal flip. At s == MAX_STRENGTH this is p=0.5, the
    #    standard setting. Boolean-mask assignment keeps it per-sample.
    flip = torch.rand(n, device=x.device) < s
    if bool(flip.any()):
        out = out.clone()
        out[flip] = torch.flip(out[flip], dims=[-1])

    # 2. Cutout: one square hole per image, zeroed across all channels. The
    #    hole is built as a coordinate mask rather than a Python loop so the
    #    cost stays negligible next to the forward pass.
    side = int(round(s * min(h, w)))
    if side >= 1:
        cy = torch.randint(0, h, (n, 1, 1), device=x.device)
        cx = torch.randint(0, w, (n, 1, 1), device=x.device)
        rows = torch.arange(h, device=x.device).view(1, h, 1)
        cols = torch.arange(w, device=x.device).view(1, 1, w)
        lo, hi = side // 2, (side + 1) // 2
        # Holes are clipped at the image edge rather than wrapped, so a hole
        # near a border covers less area - the same behavior as torchvision's
        # RandomErasing and standard cutout implementations.
        hole = (
            (rows >= cy - lo) & (rows < cy + hi) & (cols >= cx - lo) & (cols < cx + hi)
        )  # (N, H, W)
        out = out.masked_fill(hole.unsqueeze(1), 0.0)

    return out
