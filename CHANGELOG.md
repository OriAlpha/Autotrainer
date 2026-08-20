# Changelog

All notable changes to autotrainer are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning follows [SemVer](https://semver.org/) (0.x: minor bumps may change APIs).

## [Unreleased]
### Fixed
- **`optimize=True` silently disabled per-epoch reshuffling.** `prepare()`
  wraps a sharded loader in `_AutoEpochDataLoader` so
  `DistributedSampler.set_epoch()` advances on its own, but the `optimize`
  loader-defaults step and the `auto_bs` sweep each build a *fresh*
  `DataLoader`, which the wrapper did not survive. `set_epoch` was then never
  called and every epoch replayed one fixed permutation on every rank —
  silently, with nothing raised. It fired on the most common configuration
  there is: multi-GPU, `optimize=True` (the default), and an untuned loader,
  since `build_loader_defaults()` only has work to do when `num_workers` is 0
  or `pin_memory` is off. The wrapper is now applied once, after every
  rebuild. Loaders the user sharded themselves are still passed through
  unwrapped, and `shuffle=False` still means a fixed order.
- **`train()` synced the GPU twice per step.** `loss.item()` was called once
  for the running total and again for the summary; each is a device sync, so
  the loop stalled twice per iteration in the entry point whose selling point
  is throughput. Read once, reused.

### Changed
- **Docs now describe the auto-managed epoch shuffling that 0.14.0 added.**
  `docs/guide/gpu-optimization.md` still showed the pre-0.14.0 log line
  (`- call autotrainer.set_epoch(loader, epoch) ...`) where the code prints
  `- epoch shuffling auto-managed`, and the guides still presented the manual
  call as required. They now say it is optional for loaders `prepare()`
  sharded, and — worth knowing — that the automatic counter *overwrites* an
  explicit epoch number, so resuming at a specific epoch needs the loader you
  sharded yourself. `set_epoch`'s own docstring says the same.
- **Removed the `suhas` and `suhas-goravale-siddaramu` optional-dependency
  groups** from `pyproject.toml`. Both were `["psutil"]` and shipped as real
  installable extras (`pip install autotrainer[suhas]`) in the published
  metadata. The `autotrainer suhas` CLI credits command is unrelated and
  stays.
- **`ThroughputMonitor` is now in `__all__`.** It was documented in
  `docs/guide/monitors.md` with a worked example while absent from `__all__`,
  which the README declares the boundary of the stable public API — so the
  docs taught something the policy called internal. Also added to the README's
  entry-point table. `tests/test_docs.py` now checks this direction too; it
  only ever verified that `__all__` members appear in the docs, never that
  documented names are exported.

## [0.16.0] - 2026-08-20
### Added
- **Hardware Telemetry Curves & Resource Diagnostics (`#hwChart`)**:
  - Live system resource meter footer with green pulse status indicator in sidebar.
  - Dedicated hardware telemetry card displaying GPU VRAM (GB / %), GPU model name, and Host RAM / CPU utilization curves over training steps and epochs.
  - REST endpoint `GET /api/hardware` with cross-platform memory and GPU detection.
- **Run Tagging, Star Favoriting & Sidebar Organization**:
  - Star favorite toggle (⭐) on individual runs with persistent metadata state.
  - Sidebar filter tabs: `[ All ]` and `[ ⭐ Starred ]` with glow accents.
  - Inline run tag chip management: `+ Tag` creator, `✕` tag pill deletion, and dynamic sidebar tag filter pills bar.
  - REST endpoints: `POST /api/runs/<run_id>/tags` and `POST /api/runs/<run_id>/favorite`.
- **Hyperparameter & Metric Parallel Coordinates Chart (`#parallelCanvas`)**:
  - High-performance Canvas multi-axis parallel coordinates visualization in Compare Mode.
  - Visualizes complex relationships and trajectories across `Learning Rate` -> `Batch Size` -> `First Loss` -> `Training Loss` -> `Validation Loss` across multiple selected runs.
- **Run Annotations & Markdown Notes Drawer (`#notesCard`)**:
  - Embedded Markdown drawer with `Edit` and live `Preview` rendering tabs.
  - Auto-persisted to both `run.json` and `<run_dir>/notes.md` on disk.
  - REST endpoint `POST /api/runs/<run_id>/notes`.
- **1-Click Run Deletion & Archive**:
  - Clean deletion of experiment runs and log directories from disk with modal confirmation.
  - REST endpoints `DELETE /api/runs/<run_id>` and `POST /api/runs/<run_id>/archive`.
- **Refined KPI Stat Cards**:
  - Displays `FIRST LOSS`, `TRAINING LOSS`, `VALIDATION LOSS`, and `EPOCHS` with glowing accents.

## [0.15.0] - 2026-08-19
### Added
- **Zero-Dependency Executive Web UI Dashboard (`autotrainer ui`)**:
  - Live interactive training execution dashboard serving on port 8501 without external heavy dependencies.
  - Multi-user workspace filtering: dynamically tags runs by user (`AUTOTRAINER_USER`, `SLURM_JOB_USER`, OS username) and formats workspace selectors (`👥 All Users (X users, Y runs)`).
  - Dynamic multi-path directory aggregation: monitor single or multiple local/shared cluster paths (`autotrainer ui ./logs /cluster/runs`) with interactive paths management modal in the sidebar.
  - Interactive dual telemetry charts (Training & Validation loss curves, Validation accuracy / metric progression) with custom hover tooltips and chart enlargement modal.
  - AI Training Doctor & Health Triage diagnostics card with real-time health badges and actionable bulleted remediation steps.
  - 1-click run renaming directly from the dashboard with directory synchronization.
- **Multi-Format Reports & Exports**:
  - **Standalone Interactive HTML Report (`.html`)**: Self-contained executive report with dark glassmorphic styling, embedded interactive Chart.js curves, triage diagnostics, KPI cards, and print/PDF optimization.
  - **Markdown Summary (`.md`)**: Clean markdown formatted telemetry and hyperparameter tables ready for GitHub PRs, Slack, or Notion.
  - **Metrics CSV (`.csv`)** and **Full Telemetry JSON (`.json`)** downloads.
- **Native ML Framework Callbacks Suite (`autotrainer.callbacks`)**:
  - `AutotrainerCallback`: General-purpose callback for custom and vanilla PyTorch loops.
  - `AutotrainerHuggingFaceCallback`: Native `transformers.TrainerCallback` integration.
  - `AutotrainerLightningCallback`: Native PyTorch Lightning callback integration.
  - `AutotrainerKerasCallback`: Native TensorFlow / Keras callback integration.
  - `autotrainer_xgboost_callback()` & `autotrainer_lightgbm_callback()`: Native boosting callbacks.
- **Native Local Experiment Trackers (`autotrainer.trackers`)**:
  - `NativeTracker`, `CSVTracker`, and `JSONLTracker` with user attribution, metadata persistence, and thread-safe streaming.
### Fixed
- **`prepare()` auto-launch no longer kills notebook kernels.** Auto-launch
  re-executes `sys.argv` once per GPU and exits the parent, but only checked
  `RANK`, `SLURM_JOB_ID` and `detect()`. In a notebook on a multi-GPU box
  `argv[0]` is the kernel launcher, so it spawned N launchers and killed the
  kernel mid-cell with nothing to diagnose; `python -m pkg` was re-executed as
  a bare file path, losing the package context. It now requires a plain
  `python script.py` invocation and explains itself when it stands down.
- **The summary reports what autotrainer did, not what it guessed.** The
  "Active Optimizations" list was derived from global torch flags and env
  vars, which cannot distinguish a setting autotrainer made from one the user
  made. It claimed a multi-worker pinned-memory DataLoader pipeline for
  loaders left at `num_workers=0`; a "Weight Decay Exclude" norm/bias
  param-group split that autotrainer has never performed, for any nonzero
  `weight_decay`; "Grad Clipping" from `optimizer.defaults["max_norm"]`, which
  no torch optimizer defines; "SLURM Node Scratch" from `SLURM_JOB_ID` rather
  than from `configure_scratch()` running; and FlashAttention/bf16, which are
  a torch default and a hardware capability. `prepare()`, `configure_nccl()`
  and `configure_scratch()` now record what they applied, and the report
  renders **Autotrainer Applied** from that record alone. Facts observed but
  not caused move to a new **Environment Detected** section. `weight_decay`
  now appears on the optimizer recipe line, reported rather than claimed.
- **`atexit` no longer destroys the process group.** Teardown ran during
  interpreter shutdown, which is unreliable and duplicates torch's own
  cleanup; the exit hook now only prints. Explicit
  `finish(cleanup_dist=True)` is unchanged.
- **A second run in one process gets its own summary.** `finish()` now
  releases the global tracker, instead of leaving a reported tracker that
  silently suppressed the next run's summary. `fit()` still emits one summary
  for the whole run rather than reporting the search and dropping the final
  training.
- **`train()` reports its LR schedule.** The scheduler inferred by `auto()`
  was never handed to the summary, so the `LR Schedule` line was blank on the
  one path that always has one.
- **`train()` dispatches by framework.** Routing probed for `.fit` / `.forward`
  attributes, so any object with a `fit()` method was sent to the sklearn
  backend and everything else fell through to the PyTorch path, failing deep
  inside it. It now uses the same module-prefix rule as `prepare()` and
  `tune()`, raises a clear `TypeError` for models it cannot route, and sends
  boosting estimators to the boosting backend's thread config.

## [0.14.0] - 2026-07-31
### Added
- **Unified `autotrainer.train()` 1-Line API.** Single-function execution across PyTorch, Scikit-Learn, XGBoost, LightGBM, CatBoost, and TensorFlow/Keras. Automatically infers loss/optimizer/scheduler, configures thread/replica batch scaling, executes training, serializes model checkpoints, and reports performance summary.
- **Model Checkpoint `save_path=` Parameter.** Added `save_path` parameter to `train()`, `fit()`, and `tune()`, enabling direct model serialization (`.pt`, `.joblib`, `.keras`, `.json`) without requiring separate `save0()` or `joblib.dump()` code lines.
- **Automatic Summary & Process Group Cleanup.** `train()`, `fit()`, and `tune()` now automatically trigger `finish(checkpoint=save_path)` on exit, eliminating separate `autotrainer.finish()` calls.
- **`autotrainer.finish(checkpoint=...)` & `SummaryTracker` API across all backends.** One-line post-training summary reporter and cleanup helper across all supported frameworks (`torch`, `sklearn`, `xgboost`/`lightgbm`, and `tensorflow`/`keras`). Displays formatted summary box with cluster hardware, topology, VRAM usage %, training throughput (samples/sec), loss metrics, active optimizations, and health triage, then automatically destroys process groups on exit.
- **Model Parameter Count & Executive Summary Formatting.** `SummaryTracker` now calculates total trainable model parameters (`Model Parameters`), formats learning rates cleanly (`lr=4.21e-05`), detects active LR schedulers (`LR Schedule`), and renders `Autotrainer Active Optimizations` cleanly after `Artifacts & Checkpoint`.
- **Top-level `autotrainer.log_epoch()` and `autotrainer.step()` functions.** Exported in `__all__` for recording epoch metrics and step losses directly into the active summary.
- **Auto-managed DDP Epoch Shuffling (`_AutoEpochDataLoader`).** `autotrainer.prepare()` now auto-wraps DataLoaders to advance `DistributedSampler.set_epoch(epoch)` automatically on every epoch loop iteration, eliminating the need for explicit `autotrainer.set_epoch()` calls.
- **Dynamic Plain-English Optimizations Catalog.** Summary engine dynamically detects active optimizations across 25 supported methods (TF32, AMP, FlashAttention SDPA, cuDNN benchmark, DDP, FSDP, CPU parameter offloading, CUDA allocator tuning, DataLoader workers, SLURM scratch, etc.) and prints human-readable explanations.
- **Framework Diagnostics in `autotrainer doctor`.** Enhanced `autotrainer doctor` to print exact version strings for installed ML libraries (`torch v2.12.1+cpu`, `sklearn v1.9.0`, `lightgbm v4.6.0`, etc.).
- **CI Example Test Job (`test-examples`).** Added automated workflow job running all 8 example scripts (`pytorch_auto`, `pytorch_fit`, `pytorch_tune`, `pytorch_ddp`, `pytorch_optimize`, `sklearn_example`, `xgboost_example`, `tensorflow_example`) on PRs and releases.
- **Internal Architecture Catalog & Market Positioning.** Documented all 25 optimization methods in `docs/internal/architecture.md` and positioning vs PyTorch Lightning/Accelerate in `docs/internal/market-positioning.md`.
- **Unit Test Coverage.** Added `tests/test_summary.py` covering `SummaryTracker`, `finish()`, and dynamic optimization text detection. Updated `pretend_cuda` fixture in `tests/conftest.py` with `max_memory_allocated`.


- **Prerelease channel.** A version that is not exactly `X.Y.Z` (an `rc`, `a`,
  `b`, `.dev`, or `.post` suffix) is now published to TestPyPI and marked as a
  prerelease on GitHub, instead of going to production PyPI. This exists so the
  SLURM validation runbook can `pip install` and exercise a real wheel before
  the version number is spent - PyPI versions are immutable, so validating
  against a production release means any defect found costs a new version. The
  routing tests for `== 'false'` rather than `== 'true'`, so only an explicit
  "this is stable" reaches PyPI; anything unexpected goes to TestPyPI.
- Release notes for a prerelease fall back to the `[Unreleased]` changelog
  section, since an RC has no section of its own. An empty or missing section
  is still a hard failure.
- `cuda-watchdog` CI job. When the self-hosted GPU runner is offline,
  `test-cuda` sits queued indefinitely and GPU coverage silently drops to zero
  while the run still looks green - `timeout-minutes` does not help, because it
  only starts counting once a runner accepts the job. The watchdog notices and
  annotates the run.
- `tests/test_docs.py`: checks that relative doc links and heading anchors
  resolve, that everything in `__all__` is mentioned in the docs, and that no
  message promises a version that has not shipped. Every one of these
  corresponds to a real defect found by hand in this release.
- `.github/scripts/` is now linted and format-checked in CI; it writes the
  release notes and was previously unchecked.

### Fixed
- The `TypeError` raised for the removed `tune(train_loader=...)`/`val_loader=`
  aliases said they "were removed in 1.0". 1.0 has not shipped - they were
  removed in 0.12. The message pointed users at a release that does not exist.
- Excluded yanked PyPI package `build==1.5.1` from `pyproject.toml` dev dependencies (`build!=1.5.1`) and updated `uv.lock`.


### Changed
- Automatically register `get_active_summary()` when backend entry points (`prepare()` and `scope()`) are invoked.
- Documentation restructured. The README was 551 lines with 59% of it under a
  single `## Use` heading nested four levels deep; it is now a ~220-line
  entry point (pitch, install, quickstart, an entry-point table, and how
  autotrainer differs from Lightning / Accelerate / Ray / Optuna), with the
  reference material moved to `docs/guide/`. Also fixes two sections titled
  "auto batch size" 244 lines apart covering different features, and a
  paragraph that continued the quickstart 220 lines after it was interrupted.
- `docs/README.md` claimed to list everything in `autotrainer.__all__` but
  omitted `TrainingMonitor` (added 0.12) and `augment_batch` (added 0.13).

## [0.13.0] - 2026-07-28
### Added
- **`metric=` on `tune()` and `fit()`: select on the number you actually care
  about.** The torch path could only ever optimize validation *loss* - it
  picked the recipe, pruned trials, early-stopped, and chose the best epoch on
  it - while the sklearn-estimator path has scored on accuracy/R² through
  `scoring=` since 0.10. Loss is a proxy, and it comes apart from the goal
  exactly where the search space is widest: a regularized recipe (label
  smoothing, augmentation) is *less confident*, so val cross-entropy can bottom
  out and start climbing from overconfidence while val accuracy is still
  improving - and a loss-driven `patience` stops there and restores a
  less-accurate checkpoint. Pass `metric="accuracy"`, `"f1"`, `"auc"`, `"r2"`,
  or a `callable(model, loader) -> float` (with `direction=` if lower is
  better). Training still always uses the loss; this changes only what runs are
  *scored* by. Default is `"loss"`, so existing behavior is unchanged.
- **`autotrainer.metrics`.** Pure-torch implementations of the built-ins, so
  scoring on a metric doesn't make scikit-learn a hard dependency. F1 is
  macro-averaged over the classes present in the targets (the one to reach for
  on imbalanced data, where accuracy flatters a majority-class predictor); AUC
  is rank-based with average ranks for ties, which a naive implementation gets
  wrong whenever a saturated model emits identical logits, and its multiclass
  one-vs-rest form ranks on softmax probabilities rather than raw logits (they
  differ: softmax's per-sample denominator makes it non-monotone in a single
  logit column). All four agree with scikit-learn to 1e-9 across 1500
  randomized cross-checks.
- **Preemption handling in `fit(checkpoint=...)`.** The resumable checkpoint
  has existed since 0.11, but nothing caught the scheduler's warning shot, so a
  job preempted mid-epoch still threw that epoch away - and a job preempted
  during phase 1 threw away the entire search. `fit()` now watches
  SIGUSR1/SIGTERM (SLURM's `--signal=B:USR1@120`) and stops at the next epoch
  boundary *after* the checkpoint is written, so the requeued job resumes at the
  following epoch. Phase 1 is journaled to `<checkpoint>.study`, so an
  interrupted search resumes with its completed trials instead of restarting.
  Both are armed only when `checkpoint=` is set.
- **Data sanity checks in `auto()` and `tune()`.** The failures that aren't the
  framework's fault and don't announce themselves: un-normalized or non-finite
  inputs (which look like a bad learning rate), class imbalance (which looks
  like a model that "works" at 95%), constant inputs or targets, and - the
  expensive one - a train/val split that overlaps, which doesn't fail at all,
  it just reports an excellent validation score that every trial then optimizes
  against. The batches were already being peeked to infer the loss, so the
  checks are nearly free, and they run *before* the LR range test and the first
  trial rather than after an allocation is spent. Overlap is exact when both
  loaders are `Subset`s of one dataset, and otherwise compares a bounded sample
  of rows byte-for-byte (skipped below 4 features, where identical rows arise
  by chance). Warnings only - nothing is changed for you. `sanity=False` turns
  them off; `tune()` also gates them on `verbose`.
- **`autotrainer.sanity`.** The checks themselves (`report()` / `overlap()`),
  returning message strings rather than printing, so they can be called
  directly against your own loaders.
- **`autotrainer.preempt`.** The watcher behind it (`watch()` / `preempted()`),
  usable in hand-written loops. The handler only sets a flag - it never raises
  or exits - so your loop chooses a stopping point where its state is
  consistent.
- **`epochs` is now a searched hyperparameter.** Training length is one of the
  largest levers on final quality, and every trial previously ran on one fixed
  horizon. Each trial now trains for its own searched budget *and* anneals its
  LR schedule over exactly that budget, so a recipe that wants a short, fast
  anneal and one that wants a long, gentle one are each evaluated the way they
  would actually run. The range is `1..epochs_per_trial`, so adding the knob
  widens *what* is searched without making any trial cost more than it did
  before; raise `epochs_per_trial` to widen it.
- **`aug_strength`: searchable image augmentation for CNNs.** A single scalar
  dials a fixed flip + cutout policy from off to standard, and it enters the
  default space only when conv layers are detected (the policy is
  image-specific, so searching it elsewhere would burn trials on a dimension
  that changes nothing). `0.0` is in range, so "no augmentation" stays a
  candidate. The winning strength carries into `fit()`'s phase-2 retrain -
  dropping it there would retrain a different, unregularized recipe.
  Augmentation is applied to inputs on the training path only; validation and
  test passes always see clean data.
- **`autotrainer.augment_batch(x, strength)`.** The primitive behind
  `aug_strength`, exported for use in hand-written loops. Pure torch (no new
  torchvision dependency), target-preserving, never mutates its input, and a
  no-op on anything that isn't a float NCHW batch - so it can sit in a loop
  that also trains tabular or text models.

### Changed
- **ASHA now prunes on normalized rungs instead of raw epoch indices.**
  Required by the `epochs` knob: with per-trial budgets, reporting at the raw
  epoch index would pit a long-budget trial (still mid-anneal at epoch 1)
  against a short-budget one that had already finished annealing, pruning the
  long trial for converging later rather than worse. Trials now report at a
  fixed number of rungs measured as fraction-of-own-schedule-completed, which
  compares like with like. Custom `pruner=` overrides are unaffected.
- **`fit()`'s held-out test number is stored under `test_score`.** It is no
  longer necessarily a loss. `test_loss` is still set as well when the metric
  is the loss, so existing readers keep working.
- **Checkpoint format version 2** (adds `metric`/`direction`). `fit()` refuses
  to resume a checkpoint written under a different metric rather than compare
  an accuracy against a stored loss and restore the wrong epoch. Version-1
  checkpoints are rejected with the usual "delete to start fresh" message.

## [0.12.0] - 2026-07-27
### Added
- **`autotrainer.train_step(model, loss_fn, xb, yb, opt, scaler=...)`.** The
  all-in-one companion to `prepare(optimize=True)`: one call per step does the
  full AMP dance in the correct order - move the batch to the model's device,
  forward under autocast (bf16 on modern GPUs, fp16 otherwise, no-op on CPU),
  compute the loss, backward + `optimizer.step()` + `scaler.update()` (a plain
  `backward()/step()` when no scaler), and zero the grads. Returns the detached
  loss for logging. Closes the last "not fully automatic" gap: `prepare()`
  can't wrap your loop, so AMP used to be a hand-written, order-sensitive
  snippet users had to copy; `train_step()` runs it for you. Touches nothing
  about your recipe (lr/loss/schedule/optimizer). `scaler` is optional (leave
  it `None` on CPU / bf16 GPUs); `autocast=False` opts the forward out of mixed
  precision while keeping the backward/step/zero bookkeeping. Backward runs
  outside autocast (the documented-correct scoping). The AMP path now also has
  end-to-end tests (CPU: loss actually decreases through the helper; `cuda`:
  the same on-device, plus a forced-enabled fp16 scaler exercising the real
  scaling arithmetic) - previously only the pieces were unit-tested.
- **`autotrainer.TrainingMonitor`: training triage - explains numerical
  failures in plain language.** The health companion to the perf monitors:
  where `BottleneckMonitor`/`ThroughputMonitor` answer "is it fast?", this
  answers "is it healthy?". Call `monitor.step(loss, model=..., optimizer=...,
  scaler=...)` once per step (after `backward()`) and it flags, **once each**,
  with the fix: non-finite loss (lr too high / bad inputs, with a concrete
  lr/10 suggestion when the optimizer is passed), loss divergence, fp16
  gradient overflow (persistent AMP-scaler backoff -> "switch to bf16 via
  `autocast_context()`"), and non-finite / spiking / vanishing gradient norms.
  Gradient checks are skipped while an AMP scaler is active (grads are still
  scaled there, so the scaler-backoff signal owns fp16 health). `report()`
  prints a one-line all-clear or a recap. Everything but `loss` is optional;
  it only *observes* - never touches lr/loss/schedule/optimizer - and is
  opt-in with zero overhead when unused. Fully unit-tested on CPU (synthetic
  losses, a fake scaler, hand-set gradients).
- **`tune()`/`fit()` search a much wider training recipe.** The default space
  now covers the LR *schedule* (`cosine`/`onecycle`/`constant`) and its warmup
  fraction, gradient clipping (`max_norm`), and - for classification -
  label smoothing, on top of the original lr / weight-decay / optimizer /
  batch-size. The default is now *task-aware*: it's built from the model and
  the inferred loss (CNNs lean on the SGD + 1cycle recipe with a higher LR
  band; everything else defaults to a lower band with cosine/constant), so
  trials are spent where they pay off. `DEFAULT_SPACE` stays as the maximal
  superset of every searchable knob. A searched `scheduler`/`grad_clip`/
  `label_smoothing` trains identically in the short trials and the phase-2
  retrain via a shared `_make_scheduler` helper. Trials are always *scored*
  with the plain unsmoothed loss so label-smoothing candidates stay
  comparable. Does NOT search the architecture (no NAS) or the loss family
  (still inferred).
- **ASHA (successive-halving) is the default tuning pruner**, replacing the
  median pruner. Most trials get a small budget and only survivors are
  promoted - the multi-fidelity strategy that keeps the now-wider search
  affordable. `min_resource=1`, so a trial is never pruned before its second
  epoch (single-epoch trials behave exactly as before). Override with
  `tune(..., pruner=...)`.
- **`lr` is coupled to `batch_size`** when both are searched: the applied LR
  is scaled toward the trial's batch (linear for SGD, square-root for
  Adam-family, relative to a batch of 32) so the search doesn't burn trials
  rediscovering that relationship. `best_params` still records the un-scaled
  searched LR. Toggle with `tune(..., lr_scaling="none")` /
  `fit(..., lr_scaling="none")`; applied consistently across the search and
  the phase-2 retrain.
- **`fit(..., test_loader=...)`: an honest held-out generalization estimate.**
  The final (best-epoch) model is scored on a loader the search never saw;
  the number is printed and stored on rank 0's
  `study.user_attrs["test_loss"]`. Guards against reading too much into a val
  loss that a wide search has implicitly optimized against. The
  `(model, best_params, study)` return shape is unchanged.
- **`prepare()` auto-launches multi-GPU workers.** On a box with ≥2 visible
  GPUs, a bare `python train.py` now distributes across all of them with no
  launcher: the first `prepare()` call detects it's a fresh parent process
  (no `RANK`/`WORLD_SIZE` env set), spawns one worker per GPU via the same
  spawn+supervisor machinery the `autotrainer run` CLI uses, and the parent
  `sys.exit`s without entering the training loop. Each worker re-enters the
  script with `RANK` set and proceeds through `prepare()` normally (DDP wrap,
  distributed sampler). Closes the second vision gap: "drop a script in and
  it auto-detects single AND multi-GPU." Spawn only fires when all three hold
  (no `RANK` set, not under SLURM, `detect()` reports `local_multi_gpu`), so
  it never double-spawns under `srun` or loops on already-launched workers.
  Opt out with `prepare(..., auto_launch=False)`. Each spawned child is
  pinned to its own GPU via per-child `CUDA_VISIBLE_DEVICES` (the torchrun
  pattern) so workers don't race on `set_device`.
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
- **`prepare()` now defaults `optimize=True`** (was `False`). On a GPU this
  turns on the free-win bundle by default — TF32, `cudnn.benchmark` for CNNs,
  `num_workers`/`pin_memory`/`persistent_workers` defaults on bare loaders,
  and AMP — without touching lr/loss/schedule/optimizer. This closes the
  "drop a script in and it runs GPU-optimized" gap: a bare
  `prepare(model, loader)` no longer leaves the GPU wins on the table. The
  bundle is a documented no-op on CPU (every flag gates on a visible CUDA
  device), so CPU-only callers are unaffected. Pass `optimize=False` to opt
  out. When the bundle fires on a GPU with AMP on, `prepare()` now also
  prints the exact two-line `autocast_context()` + `GradScaler()` snippet to
  add to the training loop (we can't wrap an arbitrary loop from inside
  `prepare`; the helpers are no-ops on CPU so the snippet is safe verbatim).
  `auto()` and `fit()` inherit the new default automatically.
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

### Fixed
- **`ThroughputMonitor` MFU computation crashed on Python 3.9:** it used
  `zip(..., strict=True)`, which is PEP 618 (3.10+ runtime syntax). The
  project's `requires-python` and ruff `target-version` both target 3.9,
  so the 3.9 CI cell raised `TypeError: zip() takes no keyword arguments`
  in the three MFU tests. The two deques are appended in lockstep
  (`step_time` + `tick` fire once per step), so the strict check was only a
  defensive assertion; replaced with a plain `zip()` plus a comment
  documenting the lockstep invariant.
- The multi-rank FSDP wrap tests (`test_fsdp_wraps_with_orig_params_over_process_group`,
  `test_fsdp_with_cpu_offload_wraps`) failed on torch >= 2.13 in CPU-gloo CI
  with "FSDP needs a non-CPU accelerator device, but no accelerator device is
  detected". On those builds FSDP() refuses even the *wrap* when no CUDA/XPU
  device is visible - and the harness deliberately hides CUDA (`CUDA_VISIBLE_DEVICES=""`)
  so two ranks don't fight over one GPU. The worker now catches that specific
  refusal and emits a `SKIP reason=fsdp-needs-accelerator` line, so the test
  skips cleanly on builds where torch forbids the CPU wrap while still running
  the real assertion on builds where torch allows it (and where a usable GPU
  exists, the existing cuda-marked full-step test covers the rest).
- **Examples now run standalone on the spawn start method (Windows/macOS):**
  every script in `examples/` wraps its executable code in a `main()` behind
  `if __name__ == "__main__":`. Without it, `prepare()`/`fit()` setting
  `num_workers > 0` (and sklearn's joblib/loky) spawned workers that
  re-imported the module and re-ran training, crashing with "DataLoader worker
  exited unexpectedly". `autotrainer run` and `srun` were already unaffected
  (they run the script as `__main__`).

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
