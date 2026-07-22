"""SLURM ergonomics - the small things that bite people on HPC clusters.

The classic SLURM footgun: every worker process writes temporary files to
``$HOME`` (NFS-mounted, shared, slow) instead of ``$TMPDIR`` (node-local,
fast, auto-cleaned at job end). For ``torch.compile`` this means the
inductor kernel cache lives on NFS - every rank rebuilds kernels on every
run, and NFS write contention can stall the whole job.

This module exposes ``node_scratch()``: the node-local directory for this
job (``$TMPDIR`` under SLURM, system temp elsewhere) plus an ``apply()``
helper that wires the obvious env vars to it and warns when the scratch
looks like it's on a network filesystem.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def is_slurm() -> bool:
    """True when running inside a SLURM allocation (``SLURM_JOB_ID`` set)."""
    return "SLURM_JOB_ID" in os.environ


def node_scratch() -> Path:
    """The node-local scratch directory for this job.

    Under SLURM: ``$TMPDIR`` (the scheduler provisions a per-job, per-node
    directory and cleans it up at job end). Outside SLURM: the system temp
    dir - no special provisioning, but still better than ``$HOME``.

    Returns a Path that exists; creates ``$TMPDIR/autotrainer`` if needed so
    multiple autotrainer artifacts don't collide with unrelated jobs.
    """
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    # Suffix with the SLURM job id when present so concurrent jobs on the
    # same node (rare but possible with shared partitions) don't collide.
    if is_slurm():
        job = os.environ["SLURM_JOB_ID"]
        scratch = Path(base) / f"autotrainer-{job}"
    else:
        scratch = Path(base) / "autotrainer"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def _looks_networked(path: Path) -> bool:
    """Heuristic: is this directory likely on a network filesystem?

    NFS, Lustre, GPFS, and Panasas are the common HPC networked filesystems.
    Checking the device via stat() doesn't reliably distinguish them, so we
    fall back to a path-based heuristic - it catches the common case where
    ``$HOME`` (or a project dir) is NFS-mounted and ``$TMPDIR`` isn't.

    Path separators are normalized so the same markers work on POSIX and
    Windows (where ``/home/x`` resolves to ``D:\\home\\x`` and would otherwise
    lose the ``/home/`` substring).
    """
    # Normalize to forward slashes so the markers below match on Windows too.
    s = str(path.resolve()).replace("\\", "/").lower()
    return any(marker in s for marker in ("/home/", "/nfs", "/scratch/lustre", "/panfs"))


def apply(warn: bool = True) -> Path:
    """Point ``TORCHINDUCTOR_CACHE_DIR`` (and friends) at node-local scratch.

    Call once at the top of your training script, before any ``torch.compile``
    or ``prepare(..., compile=True)``. Returns the scratch Path so you can use
    it for your own checkpoints too.

    Args:
        warn: when True (default), print a warning if the resolved scratch
            looks like it's on a network filesystem (NFS/Lustre/GPFS/Panasas).
            Those work but are slow for the write-heavy kernel cache.

    Returns:
        The node-local scratch :class:`~pathlib.Path`.
    """
    from .utils import print0

    scratch = node_scratch()

    # torch.compile / inductor cache - the main thing that should NOT hit NFS.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(scratch / "inductor"))

    if warn:
        # Warn about TMPDIR itself, since that's what node_scratch() used.
        base = os.environ.get("TMPDIR") or tempfile.gettempdir()
        if _looks_networked(Path(base)):
            print0(
                f"[autotrainer] slurm: scratch {base} looks like a network "
                "filesystem (NFS/Lustre/GPFS) - torch.compile kernels and "
                "temp files will be slow. Set $TMPDIR to a node-local path "
                "(e.g. /tmp or a --gres scratch allocation) if possible."
            )
        elif is_slurm():
            print0(f"[autotrainer] slurm: using node-local scratch {scratch}")

    return scratch
