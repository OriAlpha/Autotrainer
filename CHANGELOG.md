# Changelog

All notable changes to autotrainer are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning follows [SemVer](https://semver.org/) (0.x: minor bumps may change APIs).

## [0.7.0] - 2026-07-08
### Added
- `autotrainer.tune(model, train_loader, val_loader)` — hyperparameter search over the
  training recipe (lr, weight decay, optimizer, batch size) using Optuna
  (TPE sampler + median pruning of bad trials).
- Custom search spaces via the `space=` argument.
- New optional dependency group: `pip install autotrainer[tune]`.
### Notes
- The user's model architecture is never modified; the input model object is
  never mutated. Every trial starts from the model's original initial weights.

## [0.6.0] - 2026-07-08
### Added
- `autotrainer.auto(model, loader)` — one-call smart defaults: infers loss
  function from target dtype/shape (CrossEntropy / BCE / MSE / Huber),
  selects the optimizer (AdamW default, SGD+momentum for conv nets), finds a
  learning rate via the LR range test, and builds a warmup+cosine schedule.
- `autotrainer.find_lr()` — standalone LR range test on a throwaway model copy.
- Weight decay automatically excluded from biases and norm parameters.
- Every automatic decision is printed with its reasoning and is overridable.
### Fixed
- Outlier detection for Huber-loss selection uses median/MAD instead of
  mean/std (heavy outliers inflate the std enough to hide themselves).

## [0.5.0] - 2026-07-08
### Added
- Worker failure handling in the local multi-GPU launcher: if one worker
  dies, the rest are terminated immediately instead of hanging on the next
  collective op. Clean Ctrl+C shutdown.
- Rank-aware utilities: `is_main()`, `print0()`, `save0()`, `barrier()`, `rank()`.
- `autocast_context()` — mixed precision (bf16 on supporting GPUs, fp16
  otherwise, no-op on CPU).
- `autotrainer doctor` CLI command — diagnoses frameworks, CUDA/NCCL, SLURM
  configuration (including ntasks/GPU mismatches), and rendezvous port.
- Test suite (pytest) and GitHub Actions CI with trusted PyPI publishing.

## [0.4.0] - 2026-07-08
### Added
- XGBoost and LightGBM backend: `prepare()` auto-sets `n_jobs` on
  sklearn-API estimators; `boost_params()` sets `nthread`/`num_threads`
  for the native training APIs.
### Fixed
- Dispatcher routes XGBoost/LightGBM models before the scikit-learn check
  (they subclass `BaseEstimator` and were previously misroutable).

## [0.3.0] - 2026-07-08
### Added
- TensorFlow/Keras backend: `autotrainer.scope()` context manager selecting
  `MirroredStrategy` (local multi-GPU) or `MultiWorkerMirroredStrategy`
  (SLURM multi-node) with auto-generated `TF_CONFIG` from the node list.
- `autotrainer.scale_batch_size()` — per-replica to global batch conversion.
### Changed
- Passing a TF model to `prepare()` now raises a helpful error pointing to
  `scope()` instead of failing obscurely.

## [0.2.0] - 2026-07-08
### Added
- scikit-learn backend: `prepare()` auto-sets `n_jobs`, including nested
  estimators (pipelines, grid search). Respects `SLURM_CPUS_PER_TASK` on
  clusters; uses `sched_getaffinity` locally so cgroup limits are honored.

## [0.1.0] - 2026-07-08
### Added
- Initial release. `autotrainer run script.py` CLI launcher and `autotrainer info`.
- Environment detection hierarchy: SLURM job -> cluster mode, multiple local
  GPUs -> DDP mode, otherwise single device.
- PyTorch backend: `prepare()` wraps the model in DistributedDataParallel,
  swaps in a DistributedSampler, and handles device placement.
- SLURM support: rank/rendezvous derivation from SLURM environment variables
  (`srun autotrainer run train.py`).
- `find_batch_size()` — double-until-OOM automatic batch size finder.
- pip packaging (`pyproject.toml`) with optional dependency groups.
