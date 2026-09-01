# Examples

Most examples below call `autotrainer.train()` — the one-line entry point.
`pytorch_optimize.py` is the other path: `prepare()` around a loop you wrote.

| File | Shows | Run with |
|---|---|---|
| `pytorch_auto.py` | train(): everything inferred — loss, optimizer, LR, schedule | `autotrainer run pytorch_auto.py` |
| `pytorch_optimize.py` | **prepare()**: your own loop, with DDP and the hardware wins (TF32, cudnn.benchmark, loader defaults, AMP) applied around it, plus `train_step()` | `autotrainer run pytorch_optimize.py` |
| `pytorch_ddp.py` | train() across GPUs with your own optimizer and loss: DDP, sampler, device placement, rank-0 checkpoint | `autotrainer run pytorch_ddp.py` |
| `tensorflow_example.py` | train() on Keras — auto strategy and global batch scaling | `autotrainer run tensorflow_example.py` |
| `sklearn_example.py` | train() on a nested estimator, auto `n_jobs` | `python sklearn_example.py` |
| `xgboost_example.py` | train() on both XGBoost APIs (sklearn estimator and native `DMatrix`) | `python xgboost_example.py` |

SLURM job scripts are in `slurm/` — submit with `sbatch slurm/<file>.sbatch`.
To check that multi-node DDP genuinely works on your cluster, submit
`slurm/validate_multinode.sbatch`: it runs `slurm/validate_multinode.py`, which
asserts the ranks landed on different nodes, that collectives cross between
them, that DDP really syncs gradients, and that the sampler shards the data -
one PASS/FAIL line each, with the whole environment captured in the same file.
Key rule for PyTorch DDP jobs: `--ntasks-per-node` must equal GPUs per node.
Run `autotrainer doctor` inside a job if anything misbehaves.
