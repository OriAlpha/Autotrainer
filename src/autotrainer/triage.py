"""Training triage - watch a loop and explain failures in plain language.

Where :class:`~autotrainer.BottleneckMonitor` and
:class:`~autotrainer.ThroughputMonitor` answer *"is it fast?"*, this monitor
answers *"is it healthy?"* - the numerical failures that silently waste a run:

    * loss goes ``NaN``/``Inf`` (usually lr too high, or bad inputs);
    * loss diverges (climbs far above its best);
    * fp16 gradient overflow (the AMP scaler keeps backing off);
    * gradients go non-finite, spike, or vanish;
    * loss plateaus (converged, or lr too low).

Each problem is reported ONCE, in plain language, with the fix - never a wall
of repeated warnings. It only *observes*: it never changes your lr, loss,
schedule, or optimizer (that stays yours, as everywhere in autotrainer).

Usage::

    mon = autotrainer.TrainingMonitor()
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        loss.backward()
        mon.step(loss, model=model, optimizer=opt, scaler=scaler)
        opt.step(); opt.zero_grad()
    mon.report()   # optional: one-line all-clear, or a recap of what fired

Everything but ``loss`` is optional - pass ``model`` for gradient checks,
``optimizer`` so a suggested lr can be concrete, and ``scaler`` for fp16
overflow detection. Reading the loss/grad scalars is a GPU sync point, so for
a hot loop where that overhead matters, call ``step`` every few iterations.

Opt-in: nothing is wired into the loop automatically, so there is zero
overhead when unused.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any


def _to_float(loss: Any) -> float | None:
    """Best-effort scalar from a tensor or number; ``None`` if not convertible."""
    if loss is None:
        return None
    try:
        if hasattr(loss, "detach"):
            loss = loss.detach()
        return float(loss)
    except (TypeError, ValueError, RuntimeError):
        return None


def _grad_global_norm(model: Any) -> float | None:
    """L2 norm over all parameter gradients, or ``None`` if none have a grad.

    Returns ``inf``/``nan`` faithfully when a gradient is non-finite (the
    caller treats that as its own diagnostic) rather than swallowing it.
    """
    try:
        import torch
    except ImportError:
        return None
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:
        return None
    norms = torch.stack([g.float().norm() for g in grads])
    return float(norms.norm())


class TrainingMonitor:
    """Watch per-step training signals and warn, once each, on trouble.

    Call :meth:`step` once per optimizer step (after ``backward()``, before
    ``zero_grad()`` so gradients are still live). Each diagnostic fires at most
    once so the log stays readable across a long run.

    Args:
        warmup: steps to observe before divergence / grad-spike / vanishing /
            plateau checks activate (early training is noisy).
        window: rolling window for the gradient-norm median.
        divergence_factor: warn when the smoothed loss exceeds this multiple of
            its best-so-far (needs a non-negative loss to be meaningful).
        spike_factor: warn when a gradient norm exceeds this multiple of the
            window median.
        plateau_patience: warn when the smoothed loss hasn't improved for this
            many steps.
        overflow_patience: warn after the fp16 scaler backs its scale off this
            many times (occasional backoff is normal; persistent backoff is not).
        vanish_threshold: warn when the window-median gradient norm falls below
            this.
    """

    def __init__(
        self,
        *,
        warmup: int = 20,
        window: int = 50,
        divergence_factor: float = 3.0,
        spike_factor: float = 20.0,
        plateau_patience: int = 200,
        overflow_patience: int = 3,
        vanish_threshold: float = 1e-7,
    ):
        if warmup < 0 or window < 1:
            raise ValueError("warmup must be >= 0 and window >= 1")
        if divergence_factor <= 1 or spike_factor <= 1:
            raise ValueError("divergence_factor and spike_factor must be > 1")
        if plateau_patience < 1 or overflow_patience < 1:
            raise ValueError("plateau_patience and overflow_patience must be >= 1")
        self._warmup = warmup
        self._window = window
        self._divergence_factor = divergence_factor
        self._spike_factor = spike_factor
        self._plateau_patience = plateau_patience
        self._overflow_patience = overflow_patience
        self._vanish_threshold = vanish_threshold

        self._ticks = 0
        self._smoothed: float | None = None
        self._best_loss = float("inf")
        self._best_tick = 0
        self._grad_norms: deque[float] = deque(maxlen=window)
        self._prev_scale: float | None = None
        self._overflow_count = 0
        self._fired: set[str] = set()
        self._messages: list[str] = []

    @property
    def diagnostics(self) -> list[str]:
        """The messages emitted so far (one per distinct issue), in order."""
        return list(self._messages)

    def _warn(self, key: str, msg: str) -> None:
        """Emit ``msg`` once for ``key`` (rank 0 only); later repeats are dropped."""
        if key in self._fired:
            return
        self._fired.add(key)
        from .utils import print0

        full = f"[autotrainer] triage: {msg}"
        print0(full)
        self._messages.append(full)

    def step(
        self,
        loss: Any,
        *,
        model: Any = None,
        optimizer: Any = None,
        scaler: Any = None,
    ) -> None:
        """Record one step's signals and warn on any newly-detected problem."""
        self._ticks += 1
        n = self._ticks
        lr = None
        if optimizer is not None:
            try:
                lr = optimizer.param_groups[0]["lr"]
            except (AttributeError, IndexError, KeyError, TypeError):
                lr = None

        self._check_loss(loss, n, lr)
        self._check_scaler(scaler)
        self._check_grads(model, scaler, n)

    def _check_loss(self, loss: Any, n: int, lr: float | None) -> None:
        val = _to_float(loss)
        if val is None:
            return
        if not math.isfinite(val):
            kind = "NaN" if math.isnan(val) else "Inf"
            hint = f"try lr={lr / 10:.1e}" if lr else "try lr/10"
            self._warn(
                "nan_loss",
                f"loss is {kind} at step {n} - lr is likely too high ({hint}), "
                "or an input/target contains NaN/Inf",
            )
            return

        self._smoothed = val if self._smoothed is None else 0.9 * self._smoothed + 0.1 * val
        if self._smoothed < self._best_loss:
            self._best_loss = self._smoothed
            self._best_tick = n

        if n <= self._warmup or self._best_loss <= 0:
            return  # divergence/plateau need a warmed-up, non-negative baseline
        if self._smoothed > self._divergence_factor * self._best_loss:
            self._warn(
                "diverging_loss",
                f"loss is diverging (best {self._best_loss:.3g} -> now {self._smoothed:.3g}) "
                "- lr may be too high; try lr/3 or add gradient clipping",
            )
        elif n - self._best_tick >= self._plateau_patience:
            self._warn(
                "plateau",
                f"loss hasn't improved for {n - self._best_tick} steps "
                "- you may have converged, or the lr is too low",
            )

    def _check_scaler(self, scaler: Any) -> None:
        if scaler is None or not getattr(scaler, "is_enabled", lambda: False)():
            return
        try:
            scale = float(scaler.get_scale())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return
        # The AMP scaler lowers its scale whenever it sees an inf/nan grad and
        # skips the step. Occasional backoff is by design; persistent backoff
        # means fp16 can't represent the gradients - bf16 is the real fix.
        if self._prev_scale is not None and scale < self._prev_scale:
            self._overflow_count += 1
        self._prev_scale = scale
        if self._overflow_count >= self._overflow_patience:
            self._warn(
                "fp16_overflow",
                f"fp16 gradient overflow (loss scale backed off to {scale:.0f} over "
                f"{self._overflow_count} skipped steps) - switch to bf16 via "
                "autotrainer.autocast_context() on Ampere+ GPUs, or lower the lr",
            )

    def _check_grads(self, model: Any, scaler: Any, n: int) -> None:
        # Skip when an AMP scaler is active: gradients are still scaled here, so
        # their magnitude is meaningless (the scaler path owns fp16 health).
        scaler_active = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()
        if model is None or scaler_active:
            return
        gnorm = _grad_global_norm(model)
        if gnorm is None:
            return
        if not math.isfinite(gnorm):
            kind = "NaN" if math.isnan(gnorm) else "Inf"
            self._warn(
                "nan_grad",
                f"gradients are {kind} at step {n} - numerical instability; "
                "lower the lr or add gradient clipping",
            )
            return
        self._grad_norms.append(gnorm)
        if len(self._grad_norms) < max(self._warmup, 1):
            return
        med = statistics.median(self._grad_norms)
        if med > 0 and gnorm > self._spike_factor * med:
            self._warn(
                "grad_spike",
                f"gradient norm spiked to {gnorm:.3g} ({gnorm / med:.0f}x the recent "
                "median) - add gradient clipping "
                "(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))",
            )
        if med < self._vanish_threshold:
            self._warn(
                "vanishing_grad",
                f"gradient norm is ~0 (median {med:.2g}) - vanishing gradients: "
                "dead units, lr too low, or a detached graph",
            )

    def report(self) -> str:
        """Print a formatted closing summary bullet point."""
        from .utils import print0

        if not self._messages:
            if self._ticks == 0:
                msg = "    • Status            : HEALTHY (Completed successfully; no anomalies detected)"
            else:
                msg = f"    • Status            : HEALTHY (No loss/gradient issues detected over {self._ticks} steps)"
        else:
            msg = (
                f"    • Status            : WARNING ({len(self._messages)} issue(s) flagged over "
                f"{self._ticks} steps - see details above)"
            )
        print0(msg)
        return msg

