"""Training-loop helpers that don't touch your hyperparameters.

The ``prepare(..., optimize=True)`` path sets up the GPU for throughput.
These helpers close the loop on the small things users forget *inside* the
training loop itself - zeroing grads with ``set_to_none=True`` (frees
memory), the ``model.train()``/``model.eval()`` guard pair, and gradient
accumulation when the effective batch is larger than the physical one.

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
