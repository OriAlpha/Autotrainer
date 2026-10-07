# Roadmap

Toward 1.0:

- **Stabilization**: the public execution-layer API is frozen as of 0.17.0 (`prepare()`, `train()`, `auto()`, `train_step()`). 1.0 blocks on physical multi-node SLURM verification using the self-checking validation suite (`examples/slurm/validate_multinode.sbatch`).
- **Unified Execution Layer (Shipped in 0.14.0 - 0.17.0)**:
  - 1-line unified `train()` API across PyTorch, Scikit-Learn, XGBoost, LightGBM, and TensorFlow/Keras.
  - Zero-dependency local Web UI (`autotrainer ui`), trackers (`CSVTracker`, `JSONLTracker`, `SummaryTracker`), and framework callbacks.
  - Hardware throughput bundle: TF32, AMP, cuDNN benchmark, `channels_last` for 2D convs, fused optimizers in `auto()`, and allocation-aware worker sizing (`SLURM_CPUS_PER_TASK`).
  - Diagnostics and cluster triage in `autotrainer doctor`.

Understanding your training run (Post-1.0 features):

- **Preflight estimation** (`doctor --profile`): dry-run a few batches, then
  report projected training time, memory headroom, and cost per GPU count —
  answer "how many GPUs do I actually need?" before burning an allocation.
- **Training cards**: every `train()` emits a reproducibility card (recipe,
  seeds, environment, loss curve) and `replay` reruns it.

Deeper SLURM ergonomics:

- **`autotrainer sbatch train.py --nodes 2 --time 4h`**: generate and submit a
  correct sbatch script (no more `--ntasks-per-node` != GPUs footguns).
- **Automatic requeue**: `autotrainer.preempt` already stops your loop cleanly
  at an epoch boundary when preempted, but resubmitting is still yours to do
  (`--requeue`, or `scontrol requeue`). Issuing it from inside the handler is
  the remaining step.

More breadth:

- **Multi-node boosting** (xgboost.dask / lightgbm.dask across a SLURM
  allocation) — currently single-node threads only.
- **More of the throughput bundle**: `channels_last_3d` for volumetric convs,
  and a measured default for `prefetch_factor`.


Open or upvote [issues](https://github.com/OriAlpha/Autotrainer/issues) to
prioritize these.
