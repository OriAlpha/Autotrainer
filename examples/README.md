# Examples

| File | Shows | Run with |
|---|---|---|
| `pytorch_ddp.py` | prepare(): DDP + sampler + device placement | `autotrainer run pytorch_ddp.py` |
| `pytorch_auto.py` | auto(): inferred loss/optimizer/LR/schedule, mixed precision, rank-0 helpers | `autotrainer run pytorch_auto.py` |
| `pytorch_tune.py` | tune(): hyperparameter search with Optuna | `python pytorch_tune.py` |
| `pytorch_fit.py` | fit(): tune the recipe, then fully train the winner | `autotrainer run pytorch_fit.py` |
| `tensorflow_scope.py` | scope(): auto strategy + global batch scaling | `autotrainer run tensorflow_scope.py` |
| `sklearn_example.py` | prepare() on nested estimators, auto n_jobs | `python sklearn_example.py` |
| `xgboost_example.py` | prepare() + boost_params() for both APIs | `python xgboost_example.py` |

SLURM job scripts are in `slurm/` — submit with `sbatch slurm/<file>.sbatch`.
Key rule for PyTorch DDP jobs: `--ntasks-per-node` must equal GPUs per node.
Run `autotrainer doctor` inside a job if anything misbehaves.
