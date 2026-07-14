"""`autotrainer doctor` - diagnose the environment before a job wastes GPU hours."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket

from .detect import detect

OK, WARN, FAIL = "[ ok ]", "[warn]", "[FAIL]"


def _check_frameworks(report: list[str]) -> None:
    found = []
    for name in ("torch", "tensorflow", "sklearn", "xgboost", "lightgbm"):
        if importlib.util.find_spec(name) is not None:
            found.append(name)
    if found:
        report.append(f"{OK} frameworks installed: {', '.join(found)}")
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
    _check_slurm(report)
    _check_port(report)

    print("\n".join(report))
    failed = any(line.startswith(FAIL) for line in report)
    msg = "issues found - fix [FAIL] items before training" if failed else "environment looks good"
    print(f"\n{msg}")
    return 1 if failed else 0
