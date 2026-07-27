"""Training-loop helpers that don't touch your hyperparameters.

The ``prepare(..., optimize=True)`` path sets up the GPU for throughput.
These helpers close the loop on the training loop itself: ``train_step`` runs
one full (optionally AMP) step in a single call, while the smaller helpers
cover the things users forget *inside* the loop - zeroing grads with
``set_to_none=True`` (frees memory), the ``model.train()``/``model.eval()``
guard pair, and gradient accumulation when the effective batch is larger than
the physical one.

None of these touch lr / loss / schedule / optimizer choice. They're pure
ergonomics so the loop you write is the loop you'd write by hand, minus the
forgettable boilerplate.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_ZERO_GRAD_SET_TO_NONE = True  # torch default since 2.0; kept explicit for clarity


def zero_grad(optimizer: Any) -> None:
    """``optimizer.zero_grad(set_to_none=True)`` - saves an alloc per step.

    The ``set_to_none=True`` flag is torch's own default as of 2.0, but
    being explicit here documents intent and works on older torch where
    the default was ``False`` (which wrote zeros into ``.grad`` every step,
    an unnecessary allocation).
    """
    try:
        optimizer.zero_grad(set_to_none=_ZERO_GRAD_SET_TO_NONE)
    except TypeError:
        # torch < 1.7: the kwarg didn't exist; fall back to the old behavior.
        optimizer.zero_grad()


@contextmanager
def eval_mode(model: Any) -> Iterator[Any]:
    """``model.eval()`` for the block, restore the prior train/eval state after.

    Forgetting to flip back to ``train()`` after a validation pass is the
    classic silent bug (dropout stays off, BN keeps frozen stats). This
    helper makes the scope explicit and restores automatically:

        with autotrainer.eval_mode(model):
            val_loss = evaluate(model, val_loader)
        # model is back in its prior mode here
    """
    import torch.nn as nn

    # DDP, FSDP, custom wrappers - anything with .train/.eval works. A bare
    # object with no train/eval interface just passes through unchanged.
    if not isinstance(model, nn.Module) and not (
        hasattr(model, "training") and hasattr(model, "eval") and hasattr(model, "train")
    ):
        yield model
        return
    was_training = getattr(model, "training", False)
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


@contextmanager
def train_mode(model: Any) -> Iterator[Any]:
    """``model.train()`` for the block, restore the prior state after. The
    mirror of :func:`eval_mode`; rarely needed but symmetrical."""
    if not (hasattr(model, "training") and hasattr(model, "train") and hasattr(model, "eval")):
        yield model
        return
    was_training = getattr(model, "training", True)
    model.train()
    try:
        yield model
    finally:
        model.train(was_training)


@contextmanager
def accumulate(
    optimizer: Any,
    *,
    steps: int = 1,
    scaler: Any = None,
) -> Iterator[Any]:
    """Gradient accumulation context.

    Run your forward+backward inside the block as normal; the optimizer
    steps once every ``steps`` micro-batches and grads are zeroed after.
    Lets the user hit a target effective batch on smaller GPUs without
    changing their lr or schedule (those are the user's to keep).

    Set ``steps > 1`` to accumulate. With ``scaler`` (a GradScaler), the
    ``scaler.step`` / ``scaler.update`` calls are handled, so AMP + grad
    accumulation work together without the user wiring the bookkeeping.

        with autotrainer.accumulate(opt, steps=4, scaler=scaler):
            for micro_xb, micro_yb in micro_batches:
                with autotrainer.autocast_context():
                    loss = loss_fn(model(micro_xb), micro_yb) / 4
                scaler.scale(loss).backward()
        # optimizer stepped once here; grads zeroed
    """
    if steps < 1:
        raise ValueError(f"accumulate(steps=...) must be >= 1, got {steps}")

    # Track whether the caller actually ran any backward() so we don't step
    # an optimizer with no grads (no-op for most optimizers, but clearer).
    state = {"count": 0}

    def _maybe_step(final: bool) -> None:
        if state["count"] == 0 and not final:
            return
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        zero_grad(optimizer)

    class _Accumulator:
        def backward(self, loss: Any) -> None:
            """Record one backward pass; step when the accumulator fills."""
            loss.backward()
            state["count"] += 1
            if state["count"] >= steps:
                _maybe_step(final=True)
                state["count"] = 0

    try:
        yield _Accumulator()
    finally:
        # Flush any remaining grads when the block exits mid-accumulation.
        if state["count"] > 0:
            _maybe_step(final=True)
            state["count"] = 0


def train_step(
    model: Any,
    loss_fn: Any,
    inputs: Any,
    targets: Any,
    optimizer: Any,
    *,
    scaler: Any = None,
    autocast: bool = True,
) -> Any:
    """Run one full training step and return the (detached) loss.

    The all-in-one companion to ``prepare(optimize=True)``. ``prepare()`` sets
    up the GPU for throughput but can't wrap *your* loop, so AMP otherwise
    lives in a hand-written, order-sensitive dance (scale -> backward ->
    step -> update -> zero) that's easy to get wrong. This does that dance for
    you, in the correct order:

      1. move ``inputs``/``targets`` to the model's device,
      2. forward under :func:`autotrainer.autocast_context` (bf16 on modern
         GPUs, fp16 otherwise, a no-op on CPU),
      3. ``loss = loss_fn(output, targets)``,
      4. backward + ``optimizer.step()`` + ``scaler.update()`` when a
         ``scaler`` is given (fp16), else a plain
         ``loss.backward(); optimizer.step()``,
      5. zero the grads (``set_to_none=True``).

    It touches nothing about your recipe - lr, loss, schedule, and optimizer
    choice are all yours; this only removes forgettable boilerplate. Backward
    runs *outside* the autocast context (the documented-correct scoping).

    Args:
        model: your ``nn.Module`` (already ``prepare()``-d / on its device).
            It should be in ``train()`` mode - this helper does not flip
            modes (use :func:`autotrainer.eval_mode` for validation).
        loss_fn: called as ``loss_fn(output, targets)``; must return a scalar.
        inputs: the model input - a tensor, or a tuple/list/dict of them
            (called with the same robust dispatch ``auto``/``fit`` use:
            ``model(**inputs)`` for a dict, ``model(*inputs)`` for a
            list/tuple, ``model(inputs)`` otherwise).
        targets: the labels passed as the loss's second argument.
        optimizer: your optimizer; stepped once, then zeroed. Untouched
            otherwise (lr/schedule are yours).
        scaler: a :func:`autotrainer.GradScaler` for fp16. Create it *once*
            before the loop (it carries scale state across steps) and pass it
            every step. Leave it ``None`` on CPU or bf16 GPUs, where no scaler
            is needed - the step still runs correctly. A disabled scaler
            (what ``GradScaler()`` returns on CPU/bf16) is also safe to pass.
        autocast: wrap the forward in autocast (default ``True``). Set
            ``False`` to run the forward in full precision while still using
            this helper for the backward/step/zero bookkeeping.

    Returns:
        The detached loss tensor, so ``loss.item()`` is safe for logging
        without retaining the graph.

    Example::

        model, loader, opt = autotrainer.prepare(model, loader, opt)
        scaler = autotrainer.GradScaler()
        for epoch in range(epochs):
            autotrainer.set_epoch(loader, epoch)
            model.train()
            for xb, yb in loader:
                loss = autotrainer.train_step(model, loss_fn, xb, yb, opt, scaler=scaler)
    """
    import contextlib

    from .utils import autocast_context, get_model_device, robust_forward, to_device

    device = get_model_device(model)
    inputs = to_device(inputs, device)
    targets = to_device(targets, device)

    ctx = autocast_context() if autocast else contextlib.nullcontext()
    with ctx:
        output = robust_forward(model, inputs)
        loss = loss_fn(output, targets)

    # Backward is intentionally OUTSIDE the autocast context - autocast is for
    # the forward only; running backward under it is a documented footgun.
    if scaler is not None:
        # A disabled scaler (CPU / bf16) makes scale() a pass-through,
        # step() a plain optimizer.step(), and update() a no-op - so this one
        # path is correct whether or not fp16 scaling is actually active.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    zero_grad(optimizer)
    return loss.detach()
