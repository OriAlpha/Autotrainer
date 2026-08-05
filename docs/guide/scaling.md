# Scaling up: launching, compile, FSDP, SLURM

## Launching

**`prepare()` auto-distributes.** On a multi-GPU box it spawns one worker per
GPU the first time it is called, so a bare `python train.py` uses every visible
GPU with no launcher:

```bash
python train.py                   # local: prepare() auto-spawns per-GPU workers
autotrainer run train.py          # equivalent, explicit (same spawn machinery)
autotrainer info                  # show what was detected
autotrainer doctor                # diagnose the environment
```

The auto-spawn only fires when all three hold: no `RANK` env var is set (so this
is a fresh parent, not a worker), you are not under SLURM, and there are ≥2 GPUs
on the node. That is what stops it double-spawning under `srun` or looping on
already-launched workers.

Pass `prepare(..., auto_launch=False)` to opt out — e.g. you are managing
processes yourself or running under your own launcher.

**On SLURM the front door stays `srun autotrainer run`**, because SLURM itself
starts one task per GPU across nodes and auto-spawn is correctly skipped there.
Inside your sbatch script:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
srun autotrainer run train.py
```

## `torch.compile` and FSDP

Two opt-ins on `prepare()`, both preserving the "don't touch hyperparameters"
contract:

```python
# torch.compile: 1.5-2x on modern torch+GPU for many models. Compiled
# BEFORE the DDP wrap (the documented-supported order; the reverse causes
# graph breaks). On compile failure (dynamic shapes, missing Triton on
# Windows) falls back to the uncompiled model with a warning.
model, loader, opt = autotrainer.prepare(model, loader, opt, compile=True)

# FSDP instead of DDP when the model is too large to replicate on every GPU.
# Shards params/grads/optimizer state across ranks. use_orig_params=True so
# your existing optimizer keeps working unchanged.
model, loader, opt = autotrainer.prepare(model, loader, opt, fsdp=True)
```

What it prints:

```
[autotrainer] optimize: TF32, torch.compile(mode=default), FSDP (hyperparameters untouched)
```

`compile_mode` takes `default` | `reduce-overhead` (CUDA graphs, fastest for
small models with static shapes) | `max-autotune` (kernel search, slow first
compile). None of these flags touch lr / loss / schedule / optimizer.

## CPU offload

Pairs with FSDP for the case where the model OOMs *even when sharded across
ranks* — move params to CPU and bring them to the GPU only for the
forward/backward:

```python
model, loader, opt = autotrainer.prepare(
    model, loader, opt, fsdp=True, cpu_offload=True
)
```

Trades throughput for memory headroom. Ignored with a warning on the DDP path or
single-process, where there is no sharding to offload from.

## CPU Workloads

Autotrainer handles CPU parallelism based on the backend framework:

* **PyTorch CPU (Multi-Node DDP via `gloo`)**: When no GPUs are present, `autotrainer` automatically selects PyTorch's `gloo` backend instead of `nccl` to enable multi-node or multi-process CPU training:
  ```bash
  #SBATCH --nodes=4
  #SBATCH --ntasks-per-node=1
  #SBATCH --cpus-per-task=32
  srun autotrainer run examples/pytorch_auto.py
  ```

* **Scikit-Learn & XGBoost / LightGBM (Single-Node Threading)**: These models parallelize using shared-memory threads (OpenMP / Joblib) within a single Python process. `autotrainer.prepare()` detects `SLURM_CPUS_PER_TASK` (or CPU count) and automatically configures `n_jobs`. No process launcher wrapper is needed:
  ```bash
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=16
  python examples/sklearn_example.py
  python examples/xgboost_example.py
  ```

## SLURM node-local scratch

The classic HPC footgun is every rank writing to `$HOME` (NFS, slow, shared)
instead of `$TMPDIR` (node-local, fast, auto-cleaned). One call at the top of
your script wires the obvious env vars and warns when the scratch looks
networked:

```python
import autotrainer
autotrainer.configure_scratch()  # sets TORCHINDUCTOR_CACHE_DIR to $TMPDIR
# ... your training script ...
```

Under SLURM this uses `$TMPDIR/autotrainer-<jobid>`; outside SLURM, the system
temp dir. `node_scratch()` returns the path if you want to write your own
checkpoints there too, and `configure_nccl()` sets the NCCL environment for
multi-node collectives.

For long runs under multi-process launches, raise the collective timeout with
`AUTOTRAINER_TIMEOUT` (seconds) — see [`.env.example`](../../.env.example).

## Next

- [One-call training](fit.md) — including surviving preemption on a requeued
  SLURM job.
- [runner-setup.md](../runner-setup.md) — self-hosted GPU CI runner setup.

