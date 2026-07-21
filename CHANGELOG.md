# Changelog

All notable changes to autotrainer are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning follows [SemVer](https://semver.org/) (0.x: minor bumps may change APIs).

## [Unreleased]
### Fixed
- CUDA device selection now gates on `torch.cuda.device_count() > 0`, not
  just `is_available()`, centralized in a new `autotrainer.cuda_device()`
  helper. The previous check was True whenever the driver was present,
  even when `CUDA_VISIBLE_DEVICES=""` hid every GPU - so `set_device(local_rank)`
  crashed with "invalid device ordinal" on driver-present, GPU-hidden
  boxes (e.g. the CPU-gloo distributed tests on a 1-GPU dev machine). All
  four device-pick sites (`prepare`, `_ensure_process_group`, `find_lr`,
  `_find_lr_synced`, `tune`) now share the one helper.

## [0.10.0] - 2026-07-16
### Changed (breaking - final API adjustments before 1.0)
- `auto()` now ALWAYS returns a 5-tuple
  `(model, dataloader, optimizer, loss_fn, scheduler)`; with
  `schedule=False` the scheduler is `None` instead of the tuple shrinking
  to 4 items. Unpack the fifth value (it was easy to forget which shape
  you were getting).
- `tune()`'s data parameters are now `train`/`val` (they take DataLoaders
  for PyTorch models and `(X, y)` tuples for estimators, so `_loader`
  names were misleading). `train_loader=`/`val_loader=` keyword calls
  still work but emit a `DeprecationWarning`; removal in 1.0. Positional
  calls are unaffected.
- The public API is now explicit: `autotrainer.__all__` lists it, and a
  deprecation policy in CONTRIBUTING.md covers everything in it from 1.0
  onward. Submodules and `_`-prefixed helpers are internal.
### Added
- `fit()` checkpoints now carry a `format_version`; incompatible or
  pre-0.10 checkpoints are rejected with a clear error instead of being
  silently misread.
- The pdoc API reference is published to GitHub Pages
  (https://orialpha.github.io/Autotrainer/) on every push to main.

## [0.9.0] - 2026-07-16
### Added
- `autotrainer.fit(model, train_loader, val_loader)`: one-call orchestrator
  that composes tuning and distribution. Phase 1 runs the Optuna recipe
  search; phase 2 retrains the winner from the model's original initial
  weights through `prepare()` (DDP + DistributedSampler when launched
  distributed) with a warmup+cosine schedule, mixed precision, and early
  stopping on the validation loss, restoring the best epoch's weights
  before returning `(model, best_params, study)`.
- Parallel hyperparameter search in `fit()`: when launched distributed,
  phase-1 trials are split across ALL ranks through a shared Optuna
  journal-file study (`study_storage=`, default
  `.autotrainer_study_<jobid>.log`), one trial per process on its own GPU.
  Samplers are seeded per rank; the winning recipe and inferred loss are
  broadcast so every rank trains the same configuration. `tune()` gained
  `storage=`/`study_name=` to join a shared study.
- `fit(checkpoint=...)`: preemption-safe checkpointing. Rank 0 atomically
  writes the full training state (current + best weights, optimizer,
  scheduler, recipe, early-stop counters) after every phase-2 epoch; if
  the file exists when `fit()` starts, the tuning phase is skipped and
  training resumes where it died - a requeued SLURM job rerunning the same
  script just continues.
- `tune()` now supports sklearn-API estimators (scikit-learn,
  XGBoost/LightGBM sklearn wrappers): pass `(X, y)` tuples instead of
  DataLoaders. Curated default search spaces ship for XGBoost, LightGBM,
  random forests/extra trees, gradient boosting, logistic regression,
  ridge/lasso/elastic-net, and SVM; other estimators take a custom
  `space=`. Scoring defaults to the estimator's own `.score()` (override
  with `scoring=`), thread counts follow the SLURM allocation, and the
  user's estimator object is never fitted or mutated. `fit()` raises a
  helpful TypeError pointing to `tune()` for non-PyTorch models.
- `autotrainer.set_epoch(loader, epoch)`: call at every epoch start so the
  `DistributedSampler` installed by `prepare()` reshuffles each epoch
  (without it, every epoch sees the same order). No-op for non-distributed
  loaders; `prepare()` now prints a reminder when it installs the sampler.
- `AUTOTRAINER_TIMEOUT` env var (seconds): overrides the
  `torch.distributed` collective timeout, for when one rank's phase runs
  much longer than the others' (e.g. an uneven trial split in `fit()`).
- Real 2-rank distributed tests (gloo on CPU) in the regular suite and CI:
  sampler sharding, LR-broadcast parity, and bit-identical `fit()` weights
  across ranks.
### Changed
- mypy now runs in `strict` mode (a 1.0 roadmap item): all public and
  internal signatures are fully annotated; optional-framework objects stay
  `Any` via `follow_imports = "skip"`.
### Fixed
- Local multi-GPU launches now rendezvous on a free OS-assigned port
  instead of always 29500, so two `autotrainer run` jobs on one machine no
  longer collide. An explicit `AUTOTRAINER_PORT` still pins the port, and
  SLURM keeps the fixed default (all nodes must agree up front).
- `_gpu_count()`: `CUDA_VISIBLE_DEVICES` now only restricts the detected
  GPU count instead of being trusted blindly - `CUDA_VISIBLE_DEVICES=0` on
  a GPU-less machine no longer reports a phantom GPU (which sent the
  launcher into single-GPU CUDA mode on CPU boxes).
- `prepare()` and `tune()` no longer discard user DataLoader settings when
  rebuilding loaders: `pin_memory` (was forced on CUDA), `timeout`,
  `worker_init_fn`, `generator`, `persistent_workers`, and `prefetch_factor`
  are now carried over. `prepare()` also honors `shuffle=False`
  (SequentialSampler) instead of always shuffling, passes through loaders
  that already have a `DistributedSampler`, and raises a clear `TypeError`
  for `batch_sampler=`/`IterableDataset` loaders (previously a crash or
  silent data duplication across ranks) - and it validates the loader
  before joining the process group, so a bad loader fails fast instead of
  hanging the other ranks.
- The inferred BCE loss is now directly usable on the user's batches: it
  accepts the integer `(N,)` targets that triggered its selection (plain
  `BCEWithLogitsLoss` requires float targets shaped like the `(N, 1)`
  logits and raised on the very next batch). Applies to `loss="bce"`
  overrides too.
- `auto()` under DDP now runs the LR range test on rank 0 only and
  broadcasts the result. Previously every rank swept independently -
  wasted work, and differently-shuffled loaders could land each rank on a
  different LR, silently desynchronizing the DDP replicas.

## [0.8.0] - 2026-07-14
### Added
- Open-source community files: `SECURITY.md` (vulnerability reporting policy),
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), GitHub issue templates
  (bug report, feature request) with a `config.yml` redirecting security
  reports, and a pull request template with a contributor checklist.
- `py.typed` marker (PEP 561) so downstream type checkers see autotrainer's
  annotations; shipped via `package-data`.
- PyPI classifiers expanded: development status (Beta), Python versions
  (3.9-3.13 + PyPy), audiences, operating systems, `Typing :: Typed`.
- README badges (CI, PyPI version, Python versions, license, ruff) and a
  Documentation section linking the API reference, examples, and env vars.
- Ruff configuration (`[tool.ruff]` in `pyproject.toml`) with lint + format
  rules, and a `.pre-commit-config.yaml` wiring `ruff check`/`ruff format`.
- mypy configuration and type annotations across the core modules
  (`auto_optim`, `tuning`, `__init__`, `torch_backend`, `boosting_backend`,
  `sklearn_backend`, `tf_backend`).
- pytest-cov with branch coverage and a 75% coverage floor (`fail_under`).
- Expanded test suite: `tune()`, `auto()`, the launcher (rendezvous env,
  single/SLURM/local-multi-GPU paths with failure handling), torch backend
  (`_dist_info`, single-device `prepare`, `find_batch_size` OOM backoff),
  sklearn nested-pipeline `n_jobs`, boosting native-object error, `doctor`
  SLURM/port diagnostics, in-process CLI coverage, and more utils tests.
  Coverage rose from ~unmeasured to 82%.
- `.env.example` documenting every environment variable autotrainer reads.
- pdoc API reference build (`docs/`) with a `docs/README.md` index.
- README "Roadmap" split into a "Release history" (shipped versions) and a
  forward-looking "Roadmap" (multi-node boosting, 1.0 stabilization, more
  schedulers) so the project reads as actively maintained.
### Changed
- CI workflow restructured into parallel `lint` / `test` / `typecheck` /
  `test-tf` / `docs` / `publish` jobs; `coverage.xml` uploaded as an artifact.
  TensorFlow is now exercised in its own single-version job.
- `dev` extra now includes `pytest-cov`, `mypy`, and `pdoc`.
- Modernized packaging metadata: PEP 639 SPDX license expression
  (`license`/`license-files`), removing the superseded `License ::` classifier.
- Fixed author name spelling (`Siidaramu` -> `Siddaramu`) in `pyproject.toml`
  and `LICENSE`.
- `CONTRIBUTING.md` rewritten: corrected dev-install extras, documented the
  ruff/mypy/coverage gates, added commit-message conventions, and linked the
  Code of Conduct and Security policy.
### Fixed
- `_infer_loss`: the binary-classification branch had an operator-precedence
  bug (`out_dim == 1 or n_classes == 2 and out_dim == 1` collapsed to
  `out_dim == 1`), so any single-output model was routed to `BCEWithLogitsLoss`
  regardless of class count. Now requires both `n_classes == 2` and
  `out_dim == 1` (regression test added).
- `tune()`: no longer leaves Optuna's log verbosity mutated after returning.
  The prior level is now saved and restored in a `finally` block, so a
  `tune(verbose=False)` call no longer silently disables INFO logging for the
  rest of the process (regression test added).

### Notes
- All edits are annotations, formatting, tests, tooling, and the one loss-
  inference fix above; no other runtime behavior changed.

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
