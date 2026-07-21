"""Dataloader bottleneck monitor - the cheapest piece of training triage.

GPU starvation is the most common silent slowdown: the GPU sits idle waiting
for the next batch while the user blames their model or their lr. This
monitor samples per-step ``data_time`` (waiting on the loader) vs
``step_time`` (forward+backward+step) and, after a warmup, prints a plain-
language warning when the loader dominates.

Usage:

    mon = autotrainer.BottleneckMonitor(warmup=10)
    for xb, yb in loader:
        with mon.data_time():
            pass  # xb, yb come from the iterator; this block measures the wait
        with mon.step_time():
            loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        mon.tick()
        if mon.should_report():
            mon.report()  # prints "dataloader bottleneck: 78% of step time" etc.

Optional - the user opts in by constructing the monitor. Nothing is wired
into the training loop automatically, so there is zero overhead when unused.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager


class BottleneckMonitor:
    """Sample data-load vs compute time and warn when the loader dominates.

    Keeps a rolling window (default last 50 steps) and only reports once
    every ``report_every`` ticks so it doesn't spam the log.
    """

    def __init__(self, warmup: int = 10, window: int = 50, report_every: int = 100):
        if warmup < 0 or window < 1 or report_every < 1:
            raise ValueError("warmup, window, report_every must be non-negative/positive")
        self._warmup = warmup
        self._window = window
        self._report_every = report_every
        self._data_times: deque[float] = deque(maxlen=window)
        self._step_times: deque[float] = deque(maxlen=window)
        self._ticks = 0
        self._last_report = -report_every  # so the first eligible tick reports
        # The current interval being timed by whichever `*_time` ctx is active.
        self._t0: float | None = None

    def _start(self) -> None:
        self._t0 = time.perf_counter()

    def _stop_and_record(self, buf: deque[float]) -> None:
        if self._t0 is None:
            return
        buf.append(time.perf_counter() - self._t0)
        self._t0 = None

    @contextmanager
    def data_time(self) -> Iterator[None]:
        """Time how long the step spent waiting for the next batch."""
        self._start()
        try:
            yield
        finally:
            self._stop_and_record(self._data_times)

    @contextmanager
    def step_time(self) -> Iterator[None]:
        """Time the forward+backward+optimizer step."""
        self._start()
        try:
            yield
        finally:
            self._stop_and_record(self._step_times)

    def tick(self) -> None:
        """Advance the step counter. Call once per training step."""
        self._ticks += 1

    def stats(self) -> dict[str, float]:
        """Return mean data/step time and the data-fraction of total step time.

        Returns zeros until enough samples are collected; safe to call at any time.
        """
        if len(self._data_times) < max(self._warmup, 1):
            return {"data_time": 0.0, "step_time": 0.0, "data_fraction": 0.0}
        n = min(len(self._data_times), len(self._step_times))
        data = list(self._data_times)[-n:]
        step = list(self._step_times)[-n:]
        mean_data = sum(data) / n
        mean_step = sum(step) / n
        total = mean_data + mean_step
        frac = mean_data / total if total > 0 else 0.0
        return {"data_time": mean_data, "step_time": mean_step, "data_fraction": frac}

    def should_report(self) -> bool:
        """True once we're past warmup and enough ticks have elapsed."""
        return (
            self._ticks >= self._warmup
            and len(self._data_times) >= max(self._warmup, 1)
            and self._ticks - self._last_report >= self._report_every
        )

    def report(self) -> str | None:
        """Print a one-line summary and return it. Returns None if not enough data."""
        from .utils import print0

        s = self.stats()
        if s["data_fraction"] == 0.0:
            return None
        pct = s["data_fraction"] * 100
        if pct >= 60:
            msg = (
                f"[autotrainer] bottleneck: dataloader is {pct:.0f}% of step time "
                f"(data {s['data_time'] * 1000:.0f}ms vs compute {s['step_time'] * 1000:.0f}ms) "
                "- raise num_workers / pin_memory / prefetch"
            )
        elif pct <= 25:
            msg = (
                f"[autotrainer] bottleneck: GPU-bound ({pct:.0f}% data, "
                f"{s['step_time'] * 1000:.0f}ms compute) - loader is not the limit"
            )
        else:
            msg = (
                f"[autotrainer] bottleneck: {pct:.0f}% data time "
                f"(data {s['data_time'] * 1000:.0f}ms vs compute {s['step_time'] * 1000:.0f}ms)"
            )
        print0(msg)
        self._last_report = self._ticks
        return msg
