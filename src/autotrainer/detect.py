"""Hardware and cluster environment detection.

Detection hierarchy:
    1. SLURM env vars present  -> cluster mode (trust the scheduler)
    2. Multiple local GPUs     -> local multi-GPU mode
    3. Otherwise               -> single device (GPU or CPU)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class Environment:
    mode: str                      # "slurm" | "local_multi_gpu" | "single"
    nnodes: int = 1
    nproc_per_node: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    gpus: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def world_size(self) -> int:
        return self.nnodes * self.nproc_per_node


def _gpu_count() -> int:
    """Count GPUs without importing torch (works pre-install too)."""
    # Respect CUDA_VISIBLE_DEVICES if set
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return 0 if cvd.strip() in ("", "-1") else len(cvd.split(","))
    try:
        import torch  # noqa: PLC0415
        return torch.cuda.device_count()
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
            )
            return len([l for l in out.stdout.splitlines() if l.startswith("GPU")])
        except Exception:
            return 0
    return 0


def _slurm_master_addr() -> str:
    """First hostname in the SLURM node list is the rendezvous master."""
    nodelist = os.environ.get("SLURM_NODELIST", "")
    if shutil.which("scontrol"):
        try:
            out = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True, text=True, timeout=10,
            )
            hosts = out.stdout.split()
            if hosts:
                return hosts[0]
        except Exception:
            pass
    # Fallback: crude parse of e.g. "node[01-04]" or "node01,node02"
    first = nodelist.split(",")[0]
    return first.split("[")[0] if "[" in first else (first or "127.0.0.1")


def detect() -> Environment:
    """Inspect the environment and return a launch plan."""
    if "SLURM_JOB_ID" in os.environ:
        nnodes = int(os.environ.get("SLURM_NNODES", "1"))
        gpus_on_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "0") or 0)
        nproc = gpus_on_node or int(os.environ.get("SLURM_NTASKS_PER_NODE", "1") or 1)
        env = Environment(
            mode="slurm",
            nnodes=nnodes,
            nproc_per_node=max(nproc, 1),
            node_rank=int(os.environ.get("SLURM_NODEID", "0")),
            master_addr=_slurm_master_addr(),
            master_port=int(os.environ.get("AUTOTRAINER_PORT", "29500")),
            gpus=gpus_on_node,
        )
        env.notes.append(f"SLURM job {os.environ['SLURM_JOB_ID']} detected")
        return env

    gpus = _gpu_count()
    if gpus > 1:
        return Environment(mode="local_multi_gpu", nproc_per_node=gpus, gpus=gpus)
    return Environment(mode="single", gpus=gpus)
