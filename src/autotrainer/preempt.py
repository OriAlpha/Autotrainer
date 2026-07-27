"""Preemption: notice the scheduler's warning shot and checkpoint before dying.

SLURM gives a job advance notice before it is preempted or requeued - with
``#SBATCH --signal=B:USR1@120`` the job gets ``SIGUSR1`` two minutes before it
is killed. ``fit()`` already writes a full resumable checkpoint after every
epoch, so the only thing missing was catching that signal: without it a job
preempted mid-epoch throws away the epoch it was in the middle of, and a job
preempted during phase 1 threw away the whole search.

This module is deliberately tiny and observation-only. The handler sets a
flag - it does not raise, exit, or unwind - so the training loop decides when
to stop, at an epoch boundary where the checkpoint is consistent. Anything
else risks dying halfway through writing state.

Usage inside a loop of your own::

    from autotrainer.preempt import watch, preempted
    watch()
    for epoch in range(epochs):
        train_one_epoch()
        save_checkpoint()
        if preempted():
            break        # requeue will resume from the checkpoint

``fit(checkpoint=...)`` does this for you.
"""

from __future__ import annotations

import signal
from typing import Any

# SIGUSR1 is what SLURM's --signal sends; SIGTERM is what scancel and most
# container runtimes send before SIGKILL. Named rather than imported because
# SIGUSR1 does not exist on Windows.
_DEFAULT_SIGNALS = ("SIGUSR1", "SIGTERM")

_flag = False
_installed: dict[Any, Any] = {}


def _handler(signum: int, frame: Any) -> None:
    global _flag
    _flag = True


def watch(signals: tuple[str, ...] = _DEFAULT_SIGNALS) -> list[str]:
    """Install the preemption handlers. Idempotent; returns what was installed.

    Silently installs nothing when it can't: signals absent on this platform
    (SIGUSR1 on Windows) are skipped, and ``signal.signal`` outside the main
    thread raises ``ValueError``, which is not a reason to fail a training run.
    A previously installed handler is remembered so :func:`restore` can put it
    back - autotrainer must not permanently take over a signal the user's
    program also handles.
    """
    installed = []
    for name in signals:
        sig = getattr(signal, name, None)
        if sig is None or sig in _installed:
            continue
        try:
            _installed[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)
        except (ValueError, OSError, RuntimeError):
            _installed.pop(sig, None)
            continue
        installed.append(name)
    return installed


def restore() -> None:
    """Put back whatever handlers were in place before :func:`watch`."""
    import contextlib

    for sig, previous in list(_installed.items()):
        with contextlib.suppress(ValueError, OSError, RuntimeError, TypeError):
            signal.signal(sig, previous)
        _installed.pop(sig, None)


def preempted() -> bool:
    """True once a watched signal has arrived. Never resets itself."""
    return _flag


def reset() -> None:
    """Clear the flag and uninstall the handlers (mainly for tests)."""
    global _flag
    _flag = False
    restore()
