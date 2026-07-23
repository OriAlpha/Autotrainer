# Changelog

All notable changes to autotrainer are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning follows [SemVer](https://semver.org/) (0.x: minor bumps may change APIs).

## [Unreleased]
### Added
- `prepare(..., static_graph=True)`: when distributed (DDP path), enables
  DDP's `static_graph=True` plus `gradient_as_bucket_view=True` - free
  wins when the computation graph is the same every iteration (static graph
  skips per-iteration graph-recording overhead after the first step;
  bucketing lowers peak memory). Both are opt-in because they have
  correctness implications when the graph genuinely changes (conditional
  execution, varying depth). No-op on single-device and the FSDP path. Does
  NOT touch lr / loss / schedule / optimizer.
- `prepare(..., find_unused_parameters=True)`: forwards DDP's
  `find_unused_parameters=True` for models that don't touch every param each
  step (e.g. conditional branches). Mutually exclusive with `static_graph`
  (torch forbids the combination; `prepare` raises a clear `ValueError`).
- `autotrainer.ThroughputMonitor`: throughput (samples/sec), peak GPU memory,
  and a rough model-FLOPS-utilization (MFU) estimate. The companion to
  `BottleneckMonitor` - where that answers "am I waiting on the loader?"
  (a ratio), this answers "how fast am I going, and is that fast relative to
  the GPU's peak?" (absolute numbers). MFU is opt-in (pass `model_flops`);
  the GPU-peak denominator comes from a small spec table (A100/H100/common
  consumer parts) or is reported as `None` for unknown hardware rather than
  invented. Bookkeeping is unit-tested on CPU; the real-GPU memory read is
  gated on the `cuda` marker. MFU is a diagnostic, not a benchmark - it
  assumes a matmul-heavy model.
- Multi-rank FSDP test coverage: the FSDP *wrap* + `use_orig_params` param-
  addressability path now runs against a real process group in
  `test_distributed.py` (CPU-gloo), closing the gap that the single-process
  no-op test left open. The full sharded fwd+bwd+step is gated on `>= 2`
  usable GPUs (torch 2.13 FSDP won't run a forward with CPU params when
  `cuda.is_available()` is True), and is left to a multi-GPU runner.
- `timeout-minutes: 30` on the `test-cuda` CI job, so an in-execution hang
  (driver crash, stuck NCCL collective, deadlocked test) fails loudly after
  30 minutes instead of running silently. Note this does not cover the
  "queued forever because the runner is offline" case - the timeout only
  starts once a runner picks the job up.
- `.github/dependabot.yml`: weekly dependency-update checks for the Python
  (`pip`) ecosystem and the GitHub Actions versions used in CI.
- `autotrainer.configure_nccl()`: sets `NCCL_SOCKET_IFNAME` to the
  default-route interface when unset, so multi-node SLURM jobs don't hang
  or crawl because NCCL guessed the wrong (or loopback) network interface.
  Non-clobbering (`setdefault` semantics, like `configure_scratch`); an
  optional `debug=True` turns on `NCCL_DEBUG=INFO` and prints a hint when
  detection fails (no `ip` binary, Windows host, stripped container) instead
  of a silent hang. Detection is isolated in `_detect_primary_interface()`
  (fails closed -> `None`); the env-var logic is fully unit-tested via
  monkeypatch. Real multi-node validation remains a 1.0-gate item.
- `scripts/provision-runner-python.ps1`: automates the self-hosted GPU
  runner's system-wide Python install (the `test-cuda` CI job's `env.PYTHON`
  points at `C:\Python<ver>`, which the `NETWORK SERVICE` runner account
  must be able to read). Idempotent: re-running with the same `-Version` is
  a no-op; changing `-Version` re-provisions and reminds the operator to
  update `env.PYTHON` in ci.yml. Requires admin + `uv`; `RUNNER_SETUP.md`
  now points at it as the recommended path.
- `prepare(auto_bs=True)` now prints a one-line note when the forward-only
  sweep is used (no `loss_fn`): "auto_bs running forward-only (no loss_fn);
  pass loss_fn for a larger batch size". Without a loss the sweep measures
  only activations+params, not grads + optimizer state, so the picked size
  is safe but smaller than a real fwd+bwd sweep would allow; the note
  surfaces why instead of leaving the user to find it in the docstring.
  Silent when a `loss_fn` is given (the sweep is then real).
### Changed
- Repository-hygiene cleanup (no behavior change, no public-API change):
    * Refreshed stale version references: `SECURITY.md` supported-versions
      table now reflects the current release line (`0.11.x`, was `0.7.x`);
      the bug-report issue template placeholder updated to `0.11.0`.
    * Normalized GitHub URL casing to the canonical `OriAlpha/Autotrainer`
      across `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `pyproject.toml`,
      and issue templates (GitHub redirects case-insensitively, but the
      canonical form is now used everywhere).
    * `RUNNER_SETUP.md` generalized: removed maintainer/machine-specific
      details (GPU model, driver version, runner name, fixed paths) so it
      reads as a reusable guide for any contributor setting up a self-hosted
      GPU runner.
    * `NEXT_STEPS.md` moved out of the published repo (it was an internal
      engineering backlog); it is no longer tracked. Maintainers keep it
      locally under the gitignored `docs/internal/`.
- `fitting.py` split: the phase-1 search and checkpoint helpers
  (`_unwrap`, `_sync_from_rank0`, `_journal_storage`, `_parallel_search`,
  `_save_checkpoint`, `_load_checkpoint`, `_CHECKPOINT_FORMAT`) moved to a
  new `_fit_search.py`, leaving `fitting.py` (431 -> 319 LOC) as the
  orchestrator + phase-2 training loop. Pure move, no behavior change; the
  names are re-exported from `fitting.py` so existing imports keep working.
  Prepares the ground for the roadmap's training-triage features by
  separating the two phases before layering more onto either.
- Optimize-path CUDA test stub consolidated: the per-attribute
  `_pretend_cuda` helper (patched `is_available`/`device_count`/`set_device`/
  `Tensor.to` individually) is replaced by a shared `pretend_cuda` conftest
  fixture that patches the *complete* public `torch.cuda` surface
  `prepare()` reads (`CUDA_OPTIMIZE_SURFACE`), on the real module object so
  torch's internal `torch.cuda` reads still resolve. A new guard test
  (`test_optimize.py::TestCudaSurfaceCoverage`) scans the source and fails
  loudly if `prepare()`/`_optimize`/`utils` reach for a public `torch.cuda`
  attribute the fixture doesn't stub - closing the "silently exercises the
  CPU path on a CPU-only box" gap that bit PR #1. No behavior change;
  pure test-hygiene improvement.

### Removed
- **Breaking:** `tune()` no longer accepts the `train_loader=`/`val_loader=`
  keyword aliases (deprecated in 0.10). Passing them now raises `TypeError`
  pointing at the replacement `train=`/`val=` names. The soak period elapsed;
  these names were misleading for estimator inputs (which take arrays, not
  loaders).

## [0.11.0] - 2026-07-22
### Added
- `prepare(model, loader, opt, optimize=True)`: the GPU optimization layer
  the original thesis promised - detect the hardware, set it up for
  throughput, **leave the user's hyperparameters alone**. When `optimize=True`
  and CUDA is available, `prepare()` applies:
    * `cudnn.benchmark = True` for CNNs (free win on fixed input shape).
    * TF32 on Ampere+ (`cuda.matmul.allow_tf32`, `cudnn.allow_tf32`) - a
      ~2-3x matmul speedup that ships disabled for legacy reproducibility
      reasons.
    * Loader defaults on bare `DataLoader(...)` calls: `num_workers`
      (capped at 8, sharded by world size), `pin_memory=True`,
      `persistent_workers=True`. User-set values are never overridden.
    * AMP implied (use `autotrainer.autocast_context()` + `GradScaler()` as
      before); pass `amp=False` to opt out.
  Every decision is printed; nothing in this path touches lr, loss,
  schedule, or optimizer choice. No-op on CPU and when `optimize=False`,
  so existing callers see no change.
- `prepare(..., auto_bs=True, loss_fn=...)`: grow the loader's batch size
  until OOM then back off one step. Uses the user's `loss_fn` for an
  accurate forward+backward measurement; without it the sweep is
  forward-only (conservative). The discovered size rebuilds the loader;
  lr and schedule are NOT changed - pair with `accumulate()` to scale the
  step to the new effective batch.
- `prepare(..., max_bs=N)`: ceiling for the `auto_bs` sweep (default 4096).
- `prepare(..., compile=True, compile_mode=)`: wrap the model with
  `torch.compile()` before any DDP wrap. Order matters - compiling the
  unwrapped module then DDP-wrapping is the documented-supported path;
  the reverse causes graph breaks on the `.module` indirection. No-op on
  CPU and torch < 2.0. On compile failure (dynamic shapes the backend
  can't handle, missing Triton on Windows, etc.) falls back to the
  uncompiled model with a warning rather than crashing the run.
  `compile_mode`: `default` | `reduce-overhead` (CUDA graphs) |
  `max-autotune` (kernel search). Does not touch lr/loss/schedule.
- `prepare(..., fsdp=True)`: wrap with `FullyShardedDataParallel` instead
  of `DDP` when distributed. FSDP shards params/grads/optimizer state
  across ranks - the path for models too large to fit on one GPU (DDP
  replicates and OOMs). Uses `use_orig_params=True` so the user's
  optimizer keeps working unchanged. On single-process (world_size==1)
  or torch < 2.0, `fsdp=True` is a no-op with a warning. Does not touch
  lr/loss/schedule/optimizer.
- `prepare(..., cpu_offload=True)`: when paired with `fsdp=True`, enables
  `CPUOffload(offload_params=True)` - moves FSDP-sharded params to CPU and
  brings them to GPU only for the forward/backward. Trades throughput for
  the ability to train models that OOM even when sharded across ranks.
  Ignored with a warning on the DDP path (no built-in CPU param offload)
  and on single-process (world_size == 1). Does not touch lr/loss/schedule.
- Training-loop helpers (`autotrainer.zero_grad`, `eval_mode`, `train_mode`,
  `accumulate`): the small things users forget inside the loop.
  `zero_grad` uses `set_to_none=True`; `eval_mode`/`train_mode` are
  context managers that restore the prior mode (kills the classic
  "forgot to flip back to train() after eval" bug); `accumulate(opt, steps=N)`
  handles gradient accumulation with optional `GradScaler` integration.
  None touch lr / loss / schedule / optimizer choice.
- `autotrainer.BottleneckMonitor`: the cheapest piece of the roadmap's
  training-triage theme. Sample per-step data-load vs compute time and
  print a plain-language warning when the dataloader dominates the GPU
  ("raise num_workers / pin_memory / prefetch"). Opt-in; zero overhead
  when not constructed.
- `autotrainer.node_scratch()` and `autotrainer.configure_scratch()`: SLURM
  node-local scratch ergonomics. `node_scratch()` returns `$TMPDIR`
  (per-job, per-node, auto-cleaned under SLURM; system temp elsewhere),
  suffixed with the SLURM job id so concurrent jobs don't collide.
  `configure_scratch()` sets `TORCHINDUCTOR_CACHE_DIR` to it (so
  `torch.compile` kernel cache doesn't hit NFS) and warns when the scratch
  looks like it's on a network filesystem (NFS/Lustre/GPFS/Panasas). Call
  once at the top of your training script, before any `torch.compile`.
- CUDA CI: new `cuda` pytest marker selects GPU-required tests; CPU jobs
  run `-m "not cuda"` to skip them cleanly. A `test-cuda` CI job runs the
  `cuda`-marked subset on a self-hosted GPU runner (`runs-on: [self-hosted,
  gpu]`), catching the class of CUDA-path bug that CPU-only CI cannot
  (the `device_count()` crash, the `_pretend_cuda` stub defect, and the
  `/nfs` PermissionError were all found by running on a real GPU). See
  `RUNNER_SETUP.md` for one-time runner registration, and `NEXT_STEPS.md`
  for the engineering backlog that came out of this work.
- `autotrainer.cuda_device()`: canonical helper for picking the CUDA
  device-or-CPU based on `device_count() > 0`. Centralizes the check
  previously duplicated (inconsistently) across `prepare`,
  `_ensure_process_group`, `find_lr`, `_find_lr_synced`, and `tune`.
### Fixed
- **Public `autotrainer.prepare()` now forwards keyword arguments to the torch
  backend.** Previously the dispatcher in `__init__.py` called the backend
  with only 3 positional args, so every `prepare(optimize=True)`,
  `prepare(compile=True)`, `prepare(fsdp=True)`, `prepare(auto_bs=True)`,
  etc. raised `TypeError: prepare() got an unexpected keyword argument
  'optimize'` - the entire Tier 1-3 feature surface was unreachable from the
  public API. The CI tests passed only because they imported the backend
  directly. Added a regression test class (`TestPublicDispatcherForwardsKwargs`)
  that imports the public package and exercises the dispatcher path users
  actually take.
- CUDA device selection now gates on `torch.cuda.device_count() > 0`, not
  just `is_available()`, centralized in the new `autotrainer.cuda_device()`
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
