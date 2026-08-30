"""Modular post-training summary, automatic tracking, and metrics reporting.

Two ways in, and they are separate on purpose:

* The module-level helpers (:func:`step`, :func:`log_epoch`, :func:`finish`)
  drive one process-global tracker, created on first use. This is the
  one-liner path the README documents, and what ``prepare()`` / ``fit()`` /
  ``train()`` feed into.
* :class:`SummaryTracker` instances are standalone - construct one and call
  its own ``step`` / ``log_epoch`` / ``report``. The module-level helpers do
  not see it, which is what you want when tracking two runs at once.

:func:`finish` releases the global tracker when it is done, so a second run
in the same process (a notebook, a test suite, ``fit()`` then ``train()``)
starts from a clean one instead of inheriting the finished run's losses,
timings and applied-optimization record.
"""

from __future__ import annotations

import atexit
import contextlib
import time
from pathlib import Path
from typing import Any

from .detect import detect
from .triage import TrainingMonitor
from .utils import print0

_ACTIVE_SUMMARY: SummaryTracker | None = None
_ATEXIT_REGISTERED = False
# Set once any report has been emitted, so a bare second finish() (or the
# atexit hook after an explicit finish()) doesn't print an empty second box.
_REPORT_EMITTED = False


class SummaryTracker:
    """Modular post-training summary tracker for autotrainer.

    Tracks duration, throughput, hardware topology, VRAM usage, initial vs final loss,
    validation metrics, and runs automated training health triage.

    Instances are independent of the process-global tracker that
    :func:`step` / :func:`log_epoch` / :func:`finish` operate on; call this
    object's own methods to drive it.
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        total_samples: int | None = None,
        batch_size: int | None = None,
        optimizer: Any | None = None,
        loss_fn: Any | None = None,
        scheduler: Any | None = None,
        user: str | None = None,
        logs_dir: str | Path | None = None,
    ) -> None:
        self.start_time = time.time()
        self.model = model
        self.total_samples = total_samples or 0
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.user = user
        self.logs_dir = logs_dir

        self.triage_mon = TrainingMonitor()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accs: list[float] = []
        self.step_count = 0
        self.reported = False
        # What autotrainer actually changed, keyed the same way as the dict
        # ``torch_backend.prepare()`` builds. The report renders the "Applied"
        # column by checking this. Empty dict when the user is running an
        # uninstrumented loop or another framework.
        self.applied: dict[str, Any] = {}
        self._native_tracker: Any = None

    def _get_native_tracker(self) -> Any:
        """The lazily-created on-disk tracker, or None when tracking is off.

        ``AUTOTRAINER_DISABLE_TRACKING=1`` opts out, which is how the test
        suite avoids littering ``logs/`` - set once in ``conftest.py`` rather
        than by sniffing ``PYTEST_CURRENT_TEST`` here, so the code path under
        test is the same one that ships.
        """
        import os

        if self.logs_dir is None and os.environ.get("AUTOTRAINER_DISABLE_TRACKING") == "1":
            return None
        if self._native_tracker is None:
            try:
                from .trackers import NativeTracker

                self._native_tracker = NativeTracker(
                    user=self.user, base_dir=self.logs_dir or "logs"
                )
            except Exception:
                pass
        return self._native_tracker

    @property
    def run_id(self) -> str:
        nt = self._get_native_tracker()
        return nt.run_id if nt else "run_active"

    def log_params(self, params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Log hyperparameter or configuration key-values."""
        self.record_applied(applied=params, **kwargs)

    def record_applied(self, applied: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Merge in optimizations autotrainer applied. Callers are the code
        that performed them (``prepare()``, ``configure_nccl()``, ...)."""
        if applied:
            self.applied.update(applied)
        self.applied.update(kwargs)
        nt = self._get_native_tracker()
        if nt:
            nt.log_params(self.applied)

    def end(self, summary_data: dict[str, Any] | None = None) -> None:
        """Alias for report() / finish()."""
        self.report()
        nt = self._get_native_tracker()
        if nt:
            nt.close()

    def step(
        self,
        loss: Any = None,
        batch_count: int = 1,
        model: Any | None = None,
        optimizer: Any | None = None,
    ) -> None:
        """Record a single step loss and increment sample counts."""
        self.step_count += 1
        if model is not None:
            self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        if self.batch_size:
            self.total_samples += self.batch_size * batch_count
        if loss is not None:
            opt = optimizer or self.optimizer
            self.triage_mon.step(loss, model=model, optimizer=opt)
            nt = self._get_native_tracker()
            if nt:
                with contextlib.suppress(Exception):
                    nt.log_step(self.step_count, {"loss": float(loss)})

    def log_epoch(
        self,
        train_loss: float | int | dict[str, Any] | None = None,
        val_loss: float | dict[str, Any] | None = None,
        val_acc: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log epoch-level metrics.

        Accepts three call shapes, kept for the callback integrations:

            log_epoch(train_loss, val_loss=..., val_acc=...)
            log_epoch({"train_loss": ..., "val_loss": ...})
            log_epoch(epoch_number, {"train_loss": ...})

        Every metric is optional. A callback that only has a validation
        number passes ``train_loss=None``, which used to reach
        ``float(None)`` and raise ``TypeError`` - so
        ``AutotrainerCallback.on_epoch_end(epoch=1, val_loss=0.5)``, using
        nothing but the signature's own defaults, crashed the training run
        it was supposed to be observing.
        """
        epoch_idx = len(self.train_losses) + 1
        metrics_to_log: dict[str, float] = {}

        if isinstance(train_loss, dict):
            metrics = train_loss
        elif isinstance(val_loss, dict):
            # Form: log_epoch(epoch_num, metrics_dict)
            if train_loss is not None:
                epoch_idx = int(train_loss)
            metrics = val_loss
        else:
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }

        # One place that reads the three metrics, so the dict and keyword
        # forms cannot drift apart - and one None guard covering all of them.
        for key, series, aliases in (
            ("train_loss", self.train_losses, ("train_loss", "loss")),
            ("val_loss", self.val_losses, ("val_loss",)),
            ("val_acc", self.val_accs, ("val_acc", "accuracy")),
        ):
            value = next((metrics[a] for a in aliases if metrics.get(a) is not None), None)
            if value is None:
                continue
            series.append(float(value))
            metrics_to_log[key] = float(value)

        nt = self._get_native_tracker()
        if nt and metrics_to_log:
            with contextlib.suppress(Exception):
                nt.log_epoch(epoch_idx, metrics_to_log)

    def report(self, checkpoint: str | Path | None = None) -> None:
        """Print formatted comprehensive training summary box on rank 0."""
        if self.reported:
            return
        self.reported = True

        try:
            import torch

            has_torch = True
        except ImportError:
            has_torch = False

        elapsed = time.time() - self.start_time
        num_epochs = len(self.train_losses)

        # Hardware & memory stats
        if has_torch and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            max_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            mem_str = (
                f"{max_mem_mb:.1f} MB / {total_vram_mb:.0f} MB "
                f"({max_mem_mb / total_vram_mb * 100:.1f}%)"
            )
        else:
            gpu_name = "CPU"
            mem_str = "N/A"

        world_size = (
            torch.distributed.get_world_size()
            if has_torch and torch.distributed.is_initialized()
            else 1
        )
        env_info = detect()

        # Loss metrics
        init_loss = self.train_losses[0] if self.train_losses else None
        final_loss = self.train_losses[-1] if self.train_losses else None
        loss_diff_pct = (
            ((final_loss - init_loss) / init_loss) * 100
            if (init_loss is not None and final_loss is not None and init_loss != 0)
            else None
        )

        # Throughput
        total_samples = self.total_samples
        throughput = total_samples / elapsed if elapsed > 0 and total_samples > 0 else 0.0

        # Model parameter count
        param_str = None
        if self.model and hasattr(self.model, "parameters"):
            try:
                num_params = sum(p.numel() for p in self.model.parameters())
                if num_params >= 1e9:
                    param_str = f"{num_params:,} ({num_params / 1e9:.2f}B params)"
                elif num_params >= 1e6:
                    param_str = f"{num_params:,} ({num_params / 1e6:.2f}M params)"
                elif num_params > 0:
                    param_str = f"{num_params:,} ({num_params / 1e3:.2f}K params)"
            except Exception:
                pass

        # Optimizer, LR & Loss name string
        opt_str = self.optimizer.__class__.__name__ if self.optimizer else "Standard"
        if self.optimizer and hasattr(self.optimizer, "param_groups"):
            group = self.optimizer.param_groups[0]
            bits = []
            lr = group.get("lr")
            if lr is not None:
                lr_str = f"{lr:.2e}" if (lr < 1e-3 or lr > 1e4) else f"{lr:.4g}"
                bits.append(f"lr={lr_str}")
            # Reported, not claimed: this is the user's setting. (The old
            # "Weight Decay Exclude" optimization line asserted a norm/bias
            # param-group split that autotrainer does not do.)
            wd = group.get("weight_decay")
            if wd:
                bits.append(f"weight_decay={wd:g}")
            if bits:
                opt_str += f" ({', '.join(bits)})"

        sched_str = self.scheduler.__class__.__name__ if self.scheduler else None
        loss_fn_str = self.loss_fn.__class__.__name__ if self.loss_fn else "Standard"

        print0("=" * 66)
        print0("               COMPREHENSIVE TRAINING SUMMARY             ")
        print0("=" * 66)
        print0("  Cluster & Hardware:")
        print0(f"    - GPU Device        : {gpu_name}")
        print0(f"    - Memory Usage      : {mem_str}")
        print0(
            f"    - Topology          : {env_info.nnodes} Node(s) | "  # noqa: E501
            f"{world_size} Worker Ranks (Mode: {env_info.mode})"
        )

        print0("")

        print0("  Recipe & Hyperparameters:")
        if param_str:
            print0(f"    - Model Parameters  : {param_str}")
        print0(f"    - Optimizer         : {opt_str}")
        if sched_str:
            print0(f"    - LR Schedule       : {sched_str}")
        if self.batch_size:
            print0(f"    - Batch Size        : {self.batch_size}")
        print0(f"    - Loss Function     : {loss_fn_str}")
        print0("")

        print0("  Performance & Speed:")
        print0(f"    - Total Duration    : {elapsed:.2f}s")
        if num_epochs > 0:
            print0(f"    - Avg Epoch Speed   : {elapsed / num_epochs:.2f}s / epoch")
        if total_samples > 0:
            print0(f"    - Total Dataset     : {total_samples:,} samples processed")
            print0(f"    - Throughput        : {throughput:,.2f} samples/sec")
        print0("")

        if init_loss is not None and final_loss is not None:
            print0("  Loss & Metrics:")
            print0(f"    - Initial Train Loss: {init_loss:.4f}")
            pct_str = f"  ({loss_diff_pct:+.2f}%)" if loss_diff_pct is not None else ""
            print0(f"    - Final Train Loss  : {final_loss:.4f}{pct_str}")
            if self.val_losses:
                best_val_loss = min(self.val_losses)
                print0(f"    - Best Val Loss     : {best_val_loss:.4f}")
            if self.val_accs:
                print0(f"    - Final Val Acc     : {self.val_accs[-1]:.1f}%")
            print0("")

        if checkpoint:
            ckpt_path = Path(checkpoint).resolve()
            print0("  Artifacts & Checkpoint:")
            print0(f"    - Saved Model       : {ckpt_path} (Rank-0 save)")
            print0("")

        applied_lines = self._applied_lines(world_size)
        if applied_lines:
            print0("  Autotrainer Applied:")
            for line in applied_lines:
                print0(f"    - {line}")
            print0("")

        detected_lines = self._detected_lines(has_torch)
        if detected_lines:
            print0("  Environment Detected:")
            for line in detected_lines:
                print0(f"    - {line}")
            print0("")

        print0("  Autotrainer Health Diagnostic:")
        self.triage_mon.report()
        print0("=" * 66)

        nt = self._get_native_tracker()
        if nt:
            try:
                nt.log_summary(
                    {
                        "duration": elapsed,
                        "epochs": num_epochs,
                        "throughput": throughput,
                        "init_loss": init_loss,
                        "final_loss": final_loss,
                        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
                        "triage_diagnostics": self.triage_mon.diagnostics,
                        "hardware": {
                            "device": gpu_name,
                            "memory": mem_str,
                            "world_size": world_size,
                        },
                    }
                )
                nt.close()
            except Exception:
                pass

    def _applied_lines(self, world_size: int) -> list[str]:
        """Render the "Applied" section from :attr:`applied` and nothing else.

        Every line here corresponds to a change autotrainer made on this run,
        recorded by the code that made it. Deliberately reads no
        ``torch.backends`` globals and no env vars: those are equally true
        when the *user* set them, and inferring from them is how this section
        used to report a DataLoader pipeline for loaders with
        ``num_workers=0``, and a "Weight Decay Exclude" param-group split that
        autotrainer has never performed.
        """
        a = self.applied
        lines: list[str] = []

        if a.get("tf32"):
            lines.append("TF32 matmul -> on (fp32 matmuls routed to Ampere+ tensor cores)")
        if a.get("cudnn_benchmark"):
            lines.append("cuDNN benchmark -> on (autotunes conv algorithms; conv layers found)")
        if a.get("amp"):
            lines.append("AMP -> on (bf16 where supported, else fp16 + GradScaler)")
        if a.get("compile"):
            lines.append(f"torch.compile -> mode={a['compile']}")

        loader_bits = []
        if a.get("num_workers") is not None:
            loader_bits.append(f"num_workers={a['num_workers']} (was 0)")
        if a.get("pin_memory"):
            loader_bits.append("pin_memory")
        if a.get("persistent_workers"):
            loader_bits.append("persistent_workers")
        if loader_bits:
            lines.append(f"DataLoader -> {', '.join(loader_bits)}")
        if a.get("batch_size"):
            old_bs, new_bs = a["batch_size"]
            lines.append(
                f"Batch size -> {old_bs} to {new_bs} (auto_bs sweep; lr and schedule unchanged)"
            )

        wrap = a.get("wrap")
        if wrap == "ddp":
            lines.append(f"DDP -> model replicated across {world_size} ranks")
        elif wrap == "fsdp":
            lines.append(f"FSDP -> params/grads/optimizer state sharded across {world_size} ranks")
        if a.get("ddp_opts"):
            lines.append(f"DDP options -> {', '.join(a['ddp_opts'])}")
        if a.get("cpu_offload"):
            lines.append("FSDP CPU offload -> params held on CPU between fwd/bwd")
        if a.get("sampler") == "distributed":
            lines.append("DistributedSampler -> installed (each rank sees a disjoint shard)")

        if a.get("nccl_ifname"):
            lines.append(f"NCCL_SOCKET_IFNAME -> set to {a['nccl_ifname']} (was unset)")
        if a.get("node_scratch"):
            lines.append(f"Node scratch -> temp/cache dirs pointed at {a['node_scratch']}")

        return lines

    def _detected_lines(self, has_torch: bool) -> list[str]:
        """Environment facts observed but *not* caused by autotrainer.

        Useful when diagnosing a slow run, which is why they are still shown -
        but in their own section, so nothing here reads as something
        autotrainer did. Anything autotrainer actually set is recorded in
        :attr:`applied` and reported above instead; the env-var entries below
        are skipped when that is the case.
        """
        import os

        lines: list[str] = []

        if has_torch:
            import torch

            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    lines.append("bf16 -> supported by this GPU")
                if getattr(torch.backends.cuda, "flash_sdp_enabled", None) and (
                    torch.backends.cuda.flash_sdp_enabled()
                ):
                    lines.append("Flash SDPA -> available (torch default; not set by autotrainer)")

        if "SLURM_CPUS_PER_TASK" in os.environ:
            lines.append(f"SLURM_CPUS_PER_TASK={os.environ['SLURM_CPUS_PER_TASK']}")
        if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            lines.append(f"PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
        if "NCCL_SOCKET_IFNAME" in os.environ and not self.applied.get("nccl_ifname"):
            lines.append(
                f"NCCL_SOCKET_IFNAME={os.environ['NCCL_SOCKET_IFNAME']} (preset; left alone)"
            )

        return lines


def get_active_summary() -> SummaryTracker:
    """Get or create the global active summary tracker."""
    global _ACTIVE_SUMMARY, _ATEXIT_REGISTERED
    if _ACTIVE_SUMMARY is None:
        _ACTIVE_SUMMARY = SummaryTracker()
    if not _ATEXIT_REGISTERED:
        atexit.register(_on_exit)
        _ATEXIT_REGISTERED = True
    return _ACTIVE_SUMMARY


def step(
    loss: Any = None,
    batch_count: int = 1,
    model: Any | None = None,
    optimizer: Any | None = None,
) -> None:
    """Record a step loss in the active summary tracker."""
    get_active_summary().step(loss=loss, batch_count=batch_count, model=model, optimizer=optimizer)


def log_epoch(
    train_loss: float,
    val_loss: float | None = None,
    val_acc: float | None = None,
) -> None:
    """Record epoch metrics in the active summary tracker."""
    get_active_summary().log_epoch(train_loss=train_loss, val_loss=val_loss, val_acc=val_acc)


def _on_exit() -> None:
    """Print the summary at interpreter exit if the user never called finish().

    Deliberately does NOT tear down the process group. ``destroy_process_group``
    during interpreter shutdown is unreliable - module globals are being torn
    down underneath NCCL, and torch registers its own cleanup anyway - so a
    hang or a spurious traceback here would be the last thing the user sees.
    Explicit ``finish(cleanup_dist=True)`` remains the way to ask for teardown.

    Reports only an existing, unreported tracker: after finish() the global is
    cleared, and creating a fresh one here would print an empty second box.
    """
    if _ACTIVE_SUMMARY is not None and not _ACTIVE_SUMMARY.reported:
        _ACTIVE_SUMMARY.report()


def finish(checkpoint: str | Path | None = None, cleanup_dist: bool = False) -> None:
    """Print the training summary, and optionally tear down process groups.

    Usage:
        autotrainer.finish(checkpoint="best_model.pt")

    Args:
        checkpoint: path to report as the saved artifact, if any.
        cleanup_dist: also call ``destroy_process_group()``. Default ``False``
            so ``fit()`` / ``tune()`` pipelines that call finish() between
            phases keep their process group alive.

    Releases the process-global tracker on the way out, so the next run in
    this process starts fresh rather than reusing a reported tracker (which
    would silently suppress its summary).
    """
    global _ACTIVE_SUMMARY, _REPORT_EMITTED

    summary = _ACTIVE_SUMMARY
    if summary is None and not _REPORT_EMITTED:
        # Nothing tracked yet and nothing reported yet: an explicit finish()
        # should still print, so materialize a tracker for it.
        summary = get_active_summary()
    if summary is not None and not summary.reported:
        summary.report(checkpoint=checkpoint)
        _REPORT_EMITTED = True

    if cleanup_dist:
        try:
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except ImportError:
            pass

    _ACTIVE_SUMMARY = None
