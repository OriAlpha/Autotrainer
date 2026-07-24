"""SLURM ergonomics - the small things that bite people on HPC clusters.

Two footguns this module covers:

The classic SLURM scratch footgun: every worker process writes temporary
files to ``$HOME`` (NFS-mounted, shared, slow) instead of ``$TMPDIR``
(node-local, fast, auto-cleaned at job end). For ``torch.compile`` this
means the inductor kernel cache lives on NFS - every rank rebuilds kernels
on every run, and NFS write contention can stall the whole job.

``node_scratch()`` returns the node-local scratch directory for this job
(``$TMPDIR`` under SLURM, system temp elsewhere); ``apply()`` (exported as
``configure_scratch``) wires the obvious env vars to it and warns when the
scratch looks like it's on a network filesystem.

The NCCL network-interface footgun: on multi-node clusters, NCCL often
can't infer the right network interface and falls back to a slow one (or
the loopback), so collectives hang or crawl. ``configure_nccl()`` detects
the non-loopback interface and sets ``NCCL_SOCKET_IFNAME`` (only when
unset, so a user-set value always wins), with an option to turn on
``NCCL_DEBUG=INFO`` for the first-run diagnosis.
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


def _detect_primary_interface() -> str | None:
    """Best-effort: name of the non-loopback interface with the default route.

    NCCL needs ``NCCL_SOCKET_IFNAME`` set to the interface the cluster
    interconnect runs on; if it can't guess, it scans them and frequently
    picks a slow one (or the loopback), so multi-node collectives hang or
    crawl. Picking the default-route interface is the common-case right
    answer.

    Tries, in order: ``ip -o -4 route show to default`` (parsed for the
    ``dev <iface>`` field), then ``ip route get 1.1.1.1`` (same field, for
    hosts whose default route is learned another way). Returns ``None`` if
    neither is available (e.g. Windows hosts, stripped containers) - the
    caller then leaves ``NCCL_SOCKET_IFNAME`` unset rather than guessing.

    Shells out via ``shutil.which`` so a missing ``ip`` fails closed (None)
    instead of raising. The shell-out is deliberately isolated here so tests
    monkeypatch this one function.
    """
    import shutil
    import subprocess

    ip = shutil.which("ip")
    if not ip:
        return None

    for argv in (
        [ip, "-o", "-4", "route", "show", "to", "default"],
        [ip, "route", "get", "1.1.1.1"],
    ):
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        except Exception:
            continue
        toks = out.stdout.split()
        for i, tok in enumerate(toks):
            if tok == "dev" and i + 1 < len(toks):
                iface = toks[i + 1]
                if iface and iface != "lo":
                    return iface
    return None


def configure_nccl(*, debug: bool = False) -> str | None:
    """Set ``NCCL_SOCKET_IFNAME`` to the default-route interface if unset.

    Multi-node SLURM jobs frequently hang or underperform because NCCL
    can't infer the right network interface and falls back to a slow one.
    Setting ``NCCL_SOCKET_IFNAME`` to the interface carrying the default
    route is the common-case fix.

    Idempotent and non-clobbering: a user-set ``NCCL_SOCKET_IFNAME`` always
    wins (``setdefault`` semantics, mirroring ``configure_scratch``). When
    detection fails (no ``ip`` binary, Windows host, stripped container) the
    var is left unset rather than guessed - NCCL's own scan is then the
    fallback, and the user can set it manually from the ``NCCL_DEBUG=INFO``
    output.

    Args:
        debug: when True, set ``NCCL_DEBUG=INFO`` so the first run of a job
            prints which interface NCCL chose. Useful for one-time diagnosis;
            leave off for steady-state runs (it's chatty).

    Returns:
        The interface name that was set (or was already set), or ``None``
        if detection failed and the var was unset.
    """
    from .utils import print0

    existing = os.environ.get("NCCL_SOCKET_IFNAME")
    if existing:
        # User knows best; don't second-guess a deliberate setting.
        return existing

    iface = _detect_primary_interface()
    if iface is not None:
        os.environ["NCCL_SOCKET_IFNAME"] = iface
        if debug:
            os.environ.setdefault("NCCL_DEBUG", "INFO")
        if is_slurm():
            print0(f"[autotrainer] slurm: NCCL_SOCKET_IFNAME={iface} (set if unset)")
        return iface

    # Detection failed. If the user asked for debug output, surface why
    # NCCL may be slow rather than let them chase a silent hang.
    if debug:
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        print0(
            "[autotrainer] slurm: could not detect the primary network "
            "interface; NCCL_SOCKET_IFNAME left unset. Set it manually if "
            "multi-node collectives are slow or hang."
        )
    return None
