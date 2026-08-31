# Examples

Every example below calls `autotrainer.train()` — the one-line entry point.
They differ in what they hand it and which framework they run.

| File | Shows | Run with |
|---|---|---|
| `pytorch_auto.py` | train(): everything inferred — loss, optimizer, LR, schedule | `autotrainer run pytorch_auto.py` |
| `pytorch_optimize.py` | train() with **your** optimizer and loss; hardware wins still applied (TF32, cudnn.benchmark, loader defaults, AMP) | `autotrainer run pytorch_optimize.py` |
| `pytorch_ddp.py` | the same call across GPUs: DDP, sampler, device placement, rank-0 checkpoint | `autotrainer run pytorch_ddp.py` |
| `tensorflow_example.py` | train() on Keras — auto strategy and global batch scaling | `autotrainer run tensorflow_example.py` |
| `sklearn_example.py` | train() on a nested estimator, auto `n_jobs` | `python sklearn_example.py` |
| `xgboost_example.py` | train() on both XGBoost APIs (sklearn estimator and native `DMatrix`) | `python xgboost_example.py` |

For your own training loop instead of `train()`, use `prepare(model, loader,
opt)` — see [the GPU optimization guide](../docs/guide/gpu-optimization.md).

SLURM job scripts are in `slurm/` — submit with `sbatch slurm/<file>.sbatch`.
Key rule for PyTorch DDP jobs: `--ntasks-per-node` must equal GPUs per node.
Run `autotrainer doctor` inside a job if anything misbehaves.
