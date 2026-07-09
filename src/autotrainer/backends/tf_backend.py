"""TensorFlow/Keras backend.

TF requires models to be *created inside* a strategy scope, so the API here
is a context manager rather than a wrap-after function:

    with autotrainer.scope():
        model = keras.Sequential([...])
        model.compile(...)
    model.fit(ds)

Strategy selection:
    SLURM multi-node   -> MultiWorkerMirroredStrategy (TF_CONFIG generated
                          from the SLURM node list)
    local multi-GPU    -> MirroredStrategy
    single device      -> get_strategy() (no-op default strategy)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess


def _slurm_hostnames() -> list[str]:
    nodelist = os.environ.get("SLURM_NODELIST", "")
    if shutil.which("scontrol"):
        try:
            out = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True, text=True, timeout=10,
            )
            hosts = out.stdout.split()
            if hosts:
                return hosts
        except Exception:
            pass
    return [h for h in nodelist.split(",") if h]


def build_tf_config(port: int = 29500) -> dict:
    """Build the TF_CONFIG dict for MultiWorkerMirroredStrategy from SLURM vars.

    One TF worker per node (each worker drives all local GPUs via NCCL).
    """
    hosts = _slurm_hostnames()
    node_rank = int(os.environ.get("SLURM_NODEID", "0"))
    return {
        "cluster": {"worker": [f"{h}:{port}" for h in hosts]},
        "task": {"type": "worker", "index": node_rank},
    }


def scope():
    """Return the right tf.distribute strategy scope for this environment."""
    import tensorflow as tf

    gpus = len(tf.config.list_physical_devices("GPU"))
    in_slurm = "SLURM_JOB_ID" in os.environ
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))

    if in_slurm and nnodes > 1:
        port = int(os.environ.get("AUTOTRAINER_PORT", "29500"))
        os.environ["TF_CONFIG"] = json.dumps(build_tf_config(port))
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print(f"[autotrainer] tf backend: MultiWorkerMirroredStrategy "
              f"({nnodes} nodes, {strategy.num_replicas_in_sync} replicas)")
    elif gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[autotrainer] tf backend: MirroredStrategy ({gpus} GPUs)")
    else:
        strategy = tf.distribute.get_strategy()
        print("[autotrainer] tf backend: default strategy (single device)")

    return strategy.scope()


def scale_batch_size(per_replica_batch: int) -> int:
    """TF splits the *global* batch across replicas; scale accordingly."""
    import tensorflow as tf

    strategy = tf.distribute.get_strategy()
    n = max(strategy.num_replicas_in_sync, 1)
    return per_replica_batch * n
