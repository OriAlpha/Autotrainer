"""`autotrainer doctor` - diagnose the environment before a job wastes GPU hours."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket

from .detect import detect

OK, WARN, FAIL = "[ ok ]", "[warn]", "[FAIL]"


def _check_frameworks(report: list[str]) -> None:
    details = []
    for name in ("torch", "tensorflow", "sklearn", "xgboost", "lightgbm"):
        if importlib.util.find_spec(name) is not None:
            try:
                mod = importlib.import_module(name)
                ver = getattr(mod, "__version__", "installed")
                details.append(f"{name} v{ver}")
            except Exception:
                details.append(name)
    if details:
        report.append(f"{OK} ML frameworks: {', '.join(details)}")
    else:
        report.append(
            f"{FAIL} no supported ML framework found "
            "(install torch, tensorflow, scikit-learn, xgboost, or lightgbm)"
        )


def _check_gpu(report: list[str]) -> None:
    try:
        import torch

        n = torch.cuda.device_count()
        if n:
            names = {torch.cuda.get_device_name(i) for i in range(n)}
            report.append(f"{OK} {n} CUDA GPU(s): {', '.join(sorted(names))}")
            if not torch.distributed.is_nccl_available():
                report.append(
                    f"{WARN} NCCL not available - multi-GPU will fall back to gloo (slow)"
                )
        else:
            report.append(f"{WARN} no CUDA GPUs visible (CPU mode)")
        return
    except ImportError:
        pass
    if shutil.which("nvidia-smi"):
        report.append(
            f"{WARN} nvidia-smi present but torch not installed - can't verify CUDA setup"
        )
    else:
        report.append(f"{WARN} no GPU tooling detected")


def _check_slurm(report: list[str]) -> None:
    if "SLURM_JOB_ID" not in os.environ:
        report.append(f"{OK} not inside a SLURM job (local mode)")
        return
    report.append(f"{OK} SLURM job {os.environ['SLURM_JOB_ID']} detected")
    if not shutil.which("scontrol"):
        report.append(f"{WARN} scontrol not on PATH - master addr will use crude nodelist parsing")
    if "SLURM_GPUS_ON_NODE" not in os.environ:
        report.append(f"{WARN} SLURM_GPUS_ON_NODE unset - did you request GPUs with --gres=gpu:N?")
    ntasks = os.environ.get("SLURM_NTASKS_PER_NODE")
    gpus = os.environ.get("SLURM_GPUS_ON_NODE")
    if ntasks and gpus and ntasks != gpus:
        report.append(
            f"{WARN} ntasks-per-node={ntasks} != gpus-on-node={gpus} - "
            "for DDP these should usually match (one task per GPU)"
        )
    if "SLURM_CPUS_PER_TASK" not in os.environ:
        report.append(
            f"{WARN} SLURM_CPUS_PER_TASK unset - the DataLoader will size its "
            "workers from the node's cores, not your allocation; "
            "set --cpus-per-task=N"
        )


def _check_cpus(report: list[str]) -> None:
    """The loader-worker budget, which is what actually starves the GPU.

    Two ways to lose here, and neither raises: too few CPUs and the GPU waits
    on the loader; workers sized from the node instead of the allocation and
    they thrash against the cgroup.
    """
    from ._optimize import _DEFAULT_NUM_WORKERS_CAP
    from .detect import detect
    from .utils import available_cpus

    env = detect()
    per_rank = available_cpus(env.nproc_per_node)
    workers = min(per_rank, _DEFAULT_NUM_WORKERS_CAP)
    source = "SLURM allocation" if os.environ.get("SLURM_CPUS_PER_TASK") else "affinity mask"
    line = f"{per_rank} CPU(s) per rank ({source}) -> num_workers={workers}"
    if per_rank <= 1:
        report.append(
            f"{WARN} {line} - one CPU cannot keep a GPU fed; "
            "raise --cpus-per-task (4-8 per GPU is a reasonable start)"
        )
    else:
        report.append(f"{OK} {line}")


def _check_scratch(report: list[str]) -> None:
    """Writing checkpoints and the compile cache to NFS is the classic
    HPC footgun - it works, it is just slow for everyone on the mount."""
    from .slurm import _looks_networked, node_scratch

    try:
        scratch = node_scratch()
    except OSError as e:
        report.append(f"{WARN} could not create a scratch dir: {e}")
        return
    if _looks_networked(scratch):
        report.append(
            f"{WARN} scratch {scratch} looks networked (NFS/Lustre/GPFS) - "
            "set TMPDIR to node-local storage before training"
        )
    else:
        report.append(f"{OK} scratch {scratch} looks node-local")


def _check_port(report: list[str]) -> None:
    port = int(os.environ.get("AUTOTRAINER_PORT", "29500"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            report.append(f"{OK} rendezvous port {port} is free")
        except OSError:
            report.append(f"{WARN} port {port} in use - set AUTOTRAINER_PORT to another value")


def run_doctor() -> int:
    env = detect()
    report: list[str] = [
        f"{OK} detected mode: {env.mode} "
        f"(nodes={env.nnodes}, procs/node={env.nproc_per_node}, world={env.world_size})"
    ]
    _check_frameworks(report)
    _check_gpu(report)
    _check_cpus(report)
    _check_slurm(report)
    _check_scratch(report)
    _check_port(report)

    print("\n".join(report))
    failed = any(line.startswith(FAIL) for line in report)
    msg = "issues found - fix [FAIL] items before training" if failed else "environment looks good"
    print(f"\n{msg}")
    return 1 if failed else 0
