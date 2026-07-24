"""Process launcher.

Spawns one worker process per device and sets the standard rendezvous
env vars (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) that
torch.distributed and other backends understand.

Under SLURM with `srun`, SLURM itself has already spawned one process
per task, so we don't spawn again - we just translate SLURM vars into
framework vars and exec the script in-place.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
import time

from .detect import Environment, detect


def _free_port() -> int:
    """Ask the OS for a free port so two local jobs on one machine never
    collide on the fixed default (29500)."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _rendezvous_env(env: Environment, rank: int, local_rank: int) -> dict[str, str]:
    e = os.environ.copy()
    e.update(
        RANK=str(rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(env.world_size),
        MASTER_ADDR=env.master_addr,
        MASTER_PORT=str(env.master_port),
        AUTOTRAINER_ACTIVE="1",
        AUTOTRAINER_MODE=env.mode,
    )
    return e


def _spawn_local_workers(script: str, script_args: list[str], env: Environment) -> int:
    """Spawn one child process per GPU for ``local_multi_gpu`` and supervise.

    Shared by ``launch()`` (the ``autotrainer run`` CLI) and ``prepare()``'s
    auto-launch path so both get identical, tested cleanup semantics. Each
    child re-executes ``[sys.executable, script, *script_args]`` with the
    rendezvous env vars set - so it re-enters the user's script top-to-bottom
    and hits ``prepare()`` again, this time with ``RANK`` set (which makes
    prepare skip re-spawning).

    Per-child ``CUDA_VISIBLE_DEVICES=<local_rank>`` pins each worker to its
    own GPU (the torchrun pattern): without it every child sees all GPUs and
    binding to the right one relies solely on ``torch.cuda.set_device``, which
    is fragile on some setups. With isolation, each child sees exactly one
    GPU so device 0 == its assigned GPU.

    Fail-fast: if any worker exits non-zero the rest are terminated
    immediately - a half-dead DDP job hangs forever on the next collective
    otherwise. Returns the failing exit code (0 if all exited cleanly, 130
    on Ctrl-C).
    """
    # All workers are on this machine, so any free port works as the
    # rendezvous - only an explicit AUTOTRAINER_PORT pins it. (SLURM keeps
    # the fixed default: every node must agree on the port without being
    # able to ask node 0 which one it picked.)
    port_override = os.environ.get("AUTOTRAINER_PORT")
    env.master_port = int(port_override) if port_override else _free_port()

    # Build the list of visible-device indices once: if the user restricted
    # CUDA_VISIBLE_DEVICES="2,3" we honour that ordering; otherwise each
    # local_rank maps to itself (0,1,2,...).
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd and cvd.strip() not in ("", "-1"):
        visible_ids = [d.strip() for d in cvd.split(",") if d.strip()]
    else:
        visible_ids = [str(i) for i in range(env.nproc_per_node)]

    procs = []
    for local_rank in range(env.nproc_per_node):
        child_env = _rendezvous_env(env, rank=local_rank, local_rank=local_rank)
        # Pin this child to a single GPU: it will only see device 0, which
        # is the physical GPU at visible_ids[local_rank].
        child_env["CUDA_VISIBLE_DEVICES"] = visible_ids[local_rank]
        p = subprocess.Popen([sys.executable, script, *script_args], env=child_env)
        procs.append(p)

    rc = 0
    try:
        while procs:
            for p in list(procs):
                ret = p.poll()
                if ret is None:
                    continue
                procs.remove(p)
                if ret != 0:
                    rc = ret
                    print(
                        f"[autotrainer] worker (pid {p.pid}) exited with "
                        f"code {ret}; terminating remaining workers"
                    )
                    for q in procs:
                        q.terminate()
                    for q in procs:
                        try:
                            q.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            q.kill()
                    return rc
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[autotrainer] interrupted; terminating workers")
        for q in procs:
            q.terminate()
        return 130
    return rc


def _run_script_inplace(script: str, script_args: list[str]) -> None:
    """Execute the user's script in the current process as __main__."""
    sys.argv = [script, *script_args]
    runpy.run_path(script, run_name="__main__")


def launch(script: str, script_args: list[str]) -> int:
    env = detect()
    print(
        f"[autotrainer] mode={env.mode} nodes={env.nnodes} "
        f"procs/node={env.nproc_per_node} world_size={env.world_size}"
    )
    for note in env.notes:
        print(f"[autotrainer] {note}")

    if env.mode == "slurm":
        # srun already gave us one process per task: translate and run.
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        os.environ.update(_rendezvous_env(env, rank, local_rank))
        _run_script_inplace(script, script_args)
        return 0

    if env.mode == "local_multi_gpu":
        return _spawn_local_workers(script, script_args, env)

    # single device: no distribution needed, just run it.
    os.environ.update(_rendezvous_env(env, rank=0, local_rank=0))
    _run_script_inplace(script, script_args)
    return 0
