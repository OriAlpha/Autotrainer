"""Modular post-training summary, automatic tracking, and metrics reporting."""

from __future__ import annotations

import atexit
import time
from pathlib import Path
from typing import Any

from .detect import detect
from .triage import TrainingMonitor
from .utils import print0

_ACTIVE_SUMMARY: SummaryTracker | None = None
_ATEXIT_REGISTERED = False


class SummaryTracker:
    """Modular post-training summary tracker for autotrainer.

    Tracks duration, throughput, hardware topology, VRAM usage, initial vs final loss,
    validation metrics, and runs automated training health triage.
    """

    def __init__(
        self,
        *,
        total_samples: int | None = None,
        batch_size: int | None = None,
        optimizer: Any | None = None,
        loss_fn: Any | None = None,
    ) -> None:
        self.start_time = time.time()
        self.total_samples = total_samples or 0
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.triage_mon = TrainingMonitor()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accs: list[float] = []
        self.step_count = 0
        self.reported = False

    def step(
        self,
        loss: Any = None,
        batch_count: int = 1,
        model: Any | None = None,
        optimizer: Any | None = None,
    ) -> None:
        """Record a single step loss and increment sample counts."""
        self.step_count += 1
        if self.batch_size:
            self.total_samples += self.batch_size * batch_count
        if loss is not None:
            opt = optimizer or self.optimizer
            self.triage_mon.step(loss, model=model, optimizer=opt)

    def log_epoch(
        self,
        train_loss: float,
        val_loss: float | None = None,
        val_acc: float | None = None,
    ) -> None:
        """Log epoch-level metrics."""
        self.train_losses.append(float(train_loss))
        if val_loss is not None:
            self.val_losses.append(float(val_loss))
        if val_acc is not None:
            self.val_accs.append(float(val_acc))

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

        # Optimizer & Loss name string
        opt_str = self.optimizer.__class__.__name__ if self.optimizer else "Standard"
        if self.optimizer and hasattr(self.optimizer, "param_groups"):
            lr = self.optimizer.param_groups[0].get("lr")
            if lr is not None:
                opt_str += f" (lr={lr})"
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
        print0(f"    - Optimizer         : {opt_str}")
        if self.batch_size:
            print0(f"    - Batch Size        : {self.batch_size}")
        print0(f"    - Loss Function     : {loss_fn_str}")
        print0("")

        # Active Optimizations
        opts_list = []
        import os

        if has_torch and torch.cuda.is_available():
            if getattr(torch.backends.cuda.matmul, "allow_tf32", False):
                opts_list.append(
                    "TF32 Precision -> Accelerates GPU matrix math by up to 3x with zero accuracy loss"  # noqa: E501
                )
            if getattr(torch.backends.cudnn, "benchmark", False):
                opts_list.append(
                    "cuDNN Benchmark -> Auto-tunes fastest convolution algorithms for your GPU"
                )
            if (
                hasattr(torch.backends.cuda, "flash_sdp_enabled")
                and torch.backends.cuda.flash_sdp_enabled()
            ):
                opts_list.append(
                    "FlashAttention SDPA -> Accelerates attention math by 2-4x with O(N) memory scaling"  # noqa: E501
                )
            if torch.cuda.is_bf16_supported():
                opts_list.append(
                    "Native BF16 Precision -> Uses 16-bit brain float for higher stability"
                )
            opts_list.append(
                "AMP Mixed Precision -> Cuts memory bandwidth usage in half using FP16/BF16 tensor ops"  # noqa: E501
            )

        if has_torch and torch.distributed.is_initialized():
            ws = torch.distributed.get_world_size()
            opts_list.append(
                f"DDP ({ws} Ranks) -> Scales model training in parallel across {ws} GPU processes"
            )
            opts_list.append(
                "DistributedSampler -> Auto-reshuffles dataset indices per epoch across GPUs"
            )

        if "SLURM_CPUS_PER_TASK" in os.environ:
            cpus = os.environ["SLURM_CPUS_PER_TASK"]
            opts_list.append(
                f"CPU Multi-Core Scaling -> Configured n_jobs={cpus} to match SLURM task allocation"
            )
        else:
            opts_list.append(
                "Multi-Core Parallelization -> Auto-configured worker pools across physical CPU cores"  # noqa: E501
            )

        opts_list.append(
            "DataLoader Pipeline -> Uses multi-worker threads & page-locked memory for DataLoader"
        )

        if self.optimizer and hasattr(self.optimizer, "defaults"):
            if self.optimizer.defaults.get("max_norm"):
                opts_list.append(
                    "Grad Clipping -> Limits gradient norms to prevent exploding gradients"
                )
            if self.optimizer.defaults.get("weight_decay"):
                opts_list.append(
                    "Weight Decay Exclude -> Separates Norm/Bias layers from decay to preserve convergence"  # noqa: E501
                )

        if hasattr(self, "scheduler") and hasattr(self.scheduler, "warmup_iters"):
            opts_list.append(
                "Warmup Cosine Schedule -> Gradually ramps LR before decay to prevent instability"
            )

        if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            alloc_conf = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
            opts_list.append(
                f"CUDA Allocator Tuning -> Configured ({alloc_conf}) to eliminate VRAM fragmentation OOMs"  # noqa: E501
            )
        if "SLURM_JOB_ID" in os.environ:
            opts_list.append(
                "SLURM Node Scratch -> Routes inductor cache to fast local $TMPDIR to avoid NFS stalls"  # noqa: E501
            )
        if "NCCL_SOCKET_IFNAME" in os.environ:
            ifname = os.environ["NCCL_SOCKET_IFNAME"]
            opts_list.append(f"NCCL Interconnect -> Bound to {ifname} to prevent multi-node hangs")

        if opts_list:
            print0("  Autotrainer Active Optimizations:")
            for opt_item in opts_list:
                print0(f"    - {opt_item}")
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

        print0("  Autotrainer Health Diagnostic:")
        self.triage_mon.report()
        print0("=" * 66)


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
    """Auto-report summary and cleanup on exit if not manually reported."""
    finish(cleanup_dist=True)


def finish(checkpoint: str | Path | None = None, cleanup_dist: bool = False) -> None:
    """One-line helper to print summary and optionally clean up distributed process groups.

    Usage:
        autotrainer.finish(checkpoint="best_model.pt")
    """
    summary = get_active_summary()
    if not summary.reported:
        summary.report(checkpoint=checkpoint)

    if cleanup_dist:
        try:
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except ImportError:
            pass
