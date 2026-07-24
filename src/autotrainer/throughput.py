"""Throughput and model-FLOPS-utilization (MFU) logging.

The companion to :class:`~autotrainer.BottleneckMonitor`. Where the
BottleneckMonitor answers *"am I waiting on the loader?"* (a ratio),
this one answers *"how fast am I going, and is that fast relative to
what the GPU can do?"* (absolute numbers). Together they cover the two
questions a user asks when training feels slow:

    * **BottleneckMonitor** - data-bound vs GPU-bound (relative).
    * **ThroughputMonitor** - samples/sec, peak GPU memory, and a rough
      model-FLOPS-utilization (MFU) estimate against the GPU's spec.

MFU is **opt-in**: the user must supply ``model_flops`` (the per-step
forward+backward FLOPs for one batch), because computing FLOPs for an
arbitrary model is a whole project of its own and any estimate we baked
in would be wrong as often as right. With it, MFU = achieved FLOPs /
advertised FLOPs, where the denominator comes from a small table of GPU
specs (extended as needed) or ``None`` for unknown hardware - reported
honestly, never invented.

Usage::

    mon = autotrainer.ThroughputMonitor(
        batch_size=32,
        model_flops=2 * 6 * 28 ** 2 * 256,   # per-step fwd+bwd for one batch
    )
    for xb, yb in loader:
        with mon.step_time():
            loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        mon.tick(batch_size)
        if mon.should_report():
            mon.report()

Optional - the user opts in by constructing the monitor. Nothing is
wired into the training loop automatically, so there is zero overhead
when unused.

A note on the MFU estimate: it assumes the model is matmul-heavy (true
for transformers and CNNs, the common case). It over-reports for
bandwidth- or memory-bound models (large embeddings, sparse ops, RNNs
with long sequences) and under-reports when the GPU is shared or
throttling. Treat it as a diagnostic, not a benchmark number.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

# Peak bf16/fp16 tensor throughput in TFLOPS for common GPUs, used as the
# MFU denominator. These are the *advertised* numbers (vendor spec sheets),
# not achievable-sustained - which is the point of MFU: even a well-tuned
# job runs at a fraction of the theoretical peak. Keep the table tiny and
# honestly sourced; unknown hardware returns None rather than a guess.
#
# Sources: NVIDIA datasheets (A100 / H100) and product pages (consumer
# Blackwell parts). `torch.cuda.get_device_properties().name` is matched
# case-insensitively against the keys below.
_GPU_TFLOPS: dict[str, float] = {
    # datacenter
    "a100": 312.0,
    "h100": 989.0,
    "h200": 989.0,
    "b200": 2250.0,
    # pro / workstation
    "l40": 181.0,
    "l40s": 366.0,
    "a10": 125.0,
    # consumer (advertised fp16 tensor; these vary by review, kept rough)
    "rtx 5070": 119.0,
    "rtx 5070 ti": 146.0,
    "rtx 5080": 200.0,
    "rtx 5090": 318.0,
    "rtx 4090": 165.0,
    "rtx 4080": 98.0,
    "rtx 3090": 71.0,
}


def _advertised_tflops() -> float | None:
    """Best-effort advertised peak (bf16/fp16) TFLOPS for the current CUDA
    device, or ``None`` if no GPU / unknown model.

    Looked up by matching ``torch.cuda.get_device_properties().name``
    case-insensitively against the table. ``None`` is reported honestly
    rather than inventing a number.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_properties(0).name.lower()
    # Longest key first so "rtx 5070 ti" beats "rtx 5070" on substring match.
    for key in sorted(_GPU_TFLOPS, key=len, reverse=True):
        if key in name:
            return _GPU_TFLOPS[key]
    return None


class ThroughputMonitor:
    """Track samples/sec, peak GPU memory, and a rough MFU estimate.

    Keeps a rolling window (default last 50 steps) of per-step timing and
    sample counts. MFU is only computed when the user passes
    ``model_flops`` (per-step fwd+bwd FLOPs for one batch) - without it
    only throughput and memory are reported, since auto-counting FLOPs
    for an arbitrary model would be unreliable.
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        model_flops: float | None = None,
        warmup: int = 10,
        window: int = 50,
        report_every: int = 100,
    ):
        if warmup < 0 or window < 1 or report_every < 1:
            raise ValueError("warmup, window, report_every must be non-negative/positive")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if model_flops is not None and model_flops <= 0:
            raise ValueError("model_flops must be positive")
        self._default_bs = batch_size
        self._model_flops = model_flops
        self._warmup = warmup
        self._window = window
        self._report_every = report_every
        self._step_times: deque[float] = deque(maxlen=window)
        self._samples: deque[int] = deque(maxlen=window)
        self._ticks = 0
        self._last_report = -report_every  # so the first eligible tick reports
        self._peak_mem_gib: float | None = None
        self._t0: float | None = None

    @contextmanager
    def step_time(self) -> Iterator[None]:
        """Time the forward+backward+optimizer step."""
        self._t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._t0 is not None:
                self._step_times.append(time.perf_counter() - self._t0)
                self._t0 = None

    def tick(self, batch_size: int | None = None) -> None:
        """Advance the step counter and record samples for this step.

        ``batch_size`` overrides the constructor default for this step
        (useful when the last batch is a remainder). If neither is set,
        the step is counted with zero samples (throughput/MFU will read
        as 0/None for it).
        """
        self._ticks += 1
        bs = batch_size if batch_size is not None else self._default_bs
        self._samples.append(bs or 0)
        # Read peak GPU memory lazily here (not in step_time) so the timing
        # window reflects only the user's compute, not our probing.
        self._poll_peak_memory()

    def _poll_peak_memory(self) -> None:
        try:
            import torch
        except ImportError:
            return
        if not torch.cuda.is_available():
            return
        # Reserved tracks the allocator's high-water mark; reset each report
        # cycle isn't appropriate here - peak across the whole window is what
        # predicts OOM.
        bytes_ = torch.cuda.max_memory_reserved()
        if bytes_ > 0:
            self._peak_mem_gib = bytes_ / (1024**3)

    def stats(self) -> dict[str, float | None]:
        """Return throughput, peak memory, and MFU (or None where unknowable).

        Returns zeros/None until enough samples are collected; safe to call
        at any time.
        """
        if len(self._step_times) < max(self._warmup, 1):
            return {"samples_per_sec": 0.0, "peak_mem_gib": self._peak_mem_gib, "mfu": None}
        total_time = sum(self._step_times)
        total_samples = sum(self._samples)
        sps = total_samples / total_time if total_time > 0 else 0.0

        mfu: float | None = None
        if self._model_flops is not None and total_time > 0 and total_samples > 0:
            # Achieved FLOPs/s = (FLOPs per sample) * (samples) / (window time).
            # model_flops is per-batch, so per-sample = model_flops / batch_size -
            # but batches can vary (remainder), so sum FLOPs per step instead.
            achieved = 0.0
            # _step_times and _samples are appended in lockstep (step_time +
            # tick are called once per step), so they're always equal length.
            # zip(strict=True) would assert that but is 3.10+ (PEP 618); this
            # project supports 3.9, so the plain zip relies on the invariant.
            for step_t, bs in zip(self._step_times, self._samples):
                if bs > 0 and step_t > 0:
                    achieved += (self._model_flops / max(self._default_bs or bs, 1)) * bs
            achieved /= total_time  # FLOPs/sec over the window
            peak = _advertised_tflops()
            if peak is not None:
                # peak is in TFLOPS; convert achieved (FLOPs/s) to TFLOPS.
                mfu = (achieved / 1e12) / peak
        return {"samples_per_sec": sps, "peak_mem_gib": self._peak_mem_gib, "mfu": mfu}

    def should_report(self) -> bool:
        """True once we're past warmup and enough ticks have elapsed."""
        return (
            self._ticks >= self._warmup
            and len(self._step_times) >= max(self._warmup, 1)
            and self._ticks - self._last_report >= self._report_every
        )

    def report(self) -> str | None:
        """Print a one-line summary and return it. Returns None if not enough data."""
        from .utils import print0

        s = self.stats()
        if s["samples_per_sec"] == 0.0:
            return None
        sps = s["samples_per_sec"]
        peak_mem = s["peak_mem_gib"]
        mfu = s["mfu"]

        parts = [f"throughput: {sps:.1f} samples/sec"]
        if peak_mem is not None:
            parts.append(f"peak GPU mem {peak_mem:.2f} GiB")
        if mfu is not None:
            # MFU is a fraction of advertised peak; present as a percentage.
            # 50%+ is excellent for real workloads; single digits often means
            # the model is small / bandwidth-bound / the batch is tiny.
            parts.append(f"MFU ~{mfu * 100:.0f}% of peak")
        msg = "[autotrainer] " + ", ".join(parts)
        print0(msg)
        self._last_report = self._ticks
        return msg
