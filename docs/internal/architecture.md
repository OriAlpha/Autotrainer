# Internal architecture reference

**Why this exists.** The published pdoc reference documents
`autotrainer.__all__` — 60 of the package's 166 callables, about 36%. The other
~106 are private helpers and unexported module functions, and until this file
they had no map anywhere. They are where the actual behaviour lives:
`_infer_loss`, `_maybe_auto_launch`, `_parallel_search`, `_journal_storage`.
Six months from now those are the ones you'd otherwise re-derive from source.

This is a navigation aid, not an API contract. Nothing here is public; anything
not in `__all__` can change without a deprecation cycle.

Counts as of 0.13.0: 25 modules, 4,649 LOC, 166 callables, 69% carrying
docstrings.

---

## The dependency spine

```
                          cli / __main__ / doctor
                                    |
   __init__.py  (dispatch by framework: torch? sklearn? tf? boosting?)
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
    fitting.py                  tuning.py                 auto_optim.py
   (fit: search+train)        (recipe search)          (infer loss/opt/sched)
        |                           |                           |
        +----------> backends/torch_backend.py <----------------+
                                    |
              _optimize · launcher · detect · utils
```

`utils.py` is imported by **14** of 25 modules — it is the floor, and a change
there touches nearly everything. After that: `tuning`, `backends.torch_backend`,
and `detect` (4 importers each).

`fitting` is imported only by `__init__` (the dispatcher) — nothing inside the
package depends on it. That is why it is safe to keep splitting it
(`_fit_search.py` was carved out of it already, and at 457 LOC it is still the
most overloaded module in the package — NEXT_STEPS item 10).

---

## Layer 1 — Foundation

### `utils.py` (227 LOC, 15 public, 0 private)

Rank-aware utilities and mixed precision. **The most depended-upon module in
the package.** Unusually, it is entirely public-named — every helper is
importable, though only 8 are in `__all__`.

The 7 that are *not* exported are the internal plumbing the backends lean on:
`cuda_device`, `get_model_device`, `to_device`, `slice_batch`, `split_xy`,
`robust_forward`, `get_batch_size`.

`robust_forward` is the one to know: it is how the package calls a user model
whose signature it does not control. `split_xy` and `slice_batch` are the
batch-shape guesswork that lets `auto()` and `tune()` peek at data without
knowing its structure.

`GradScaler()` is a factory, not a class — it prefers
`torch.amp.GradScaler("cuda")` and falls back to the deprecated
`torch.cuda.amp.GradScaler` on older torch.

### `loop.py` (236 LOC)

`zero_grad`, `eval_mode`, `train_mode`, `accumulate`, `train_step`. All five
public and exported. Contract: none of them touches lr, loss, schedule, or
optimizer choice. `train_step` is the newest (0.12) and the only one that runs
a whole step.

### `detect.py` (109 LOC)

Hardware and cluster detection. `detect()` returns an `Environment`; everything
else decides *what mode we are in*. `_gpu_count` and `_slurm_master_addr` are
the two probes that matter, and `Environment.world_size` is the derived number
every launch path keys off.

Imported by `cli`, `doctor`, `launcher`, `torch_backend` — the four places that
need to know what machine they're on.

---

## Layer 2 — Backends

Framework dispatch happens in `__init__.py`, which sniffs `type(model).__module__`
and routes. Each backend exposes a `prepare()`.

### `backends/torch_backend.py` (598 LOC — the largest module)

Six private helpers doing the real work:

| Function | Owns |
|---|---|
| `_dist_info` | who am I in the world (rank, local rank, world size) |
| `_ensure_process_group` | idempotent `init_process_group` |
| `_maybe_auto_launch` | the auto-spawn decision |
| `_ddp_kwargs` | DDP construction args |
| `_loader_kwargs` | sampler + worker settings |
| `_shard_loader` | rebuild the DataLoader with a `DistributedSampler` |

`_maybe_auto_launch` is the subtle one. Auto-spawn fires only when all three
hold: no `RANK` set, not under SLURM, and ≥2 GPUs. Get that wrong and you
either double-spawn under `srun` or infinitely recurse on already-launched
workers.

### `backends/tf_backend.py` · `sklearn_backend.py` · `boosting_backend.py`

Much thinner (90 / 50 / 63 LOC). TF owns the strategy scope and
`scale_batch_size`; sklearn is a near-passthrough; boosting sets thread counts
via `boost_params`.

---

## Layer 3 — Support

### `_optimize.py` (148 LOC)

Everything `prepare(optimize=True)` flips. `apply_gpu_flags` (TF32,
`cudnn.benchmark`), `build_loader_defaults` (workers/pin/persistent), and
`summarize` — which produces the "here is what I changed" line. `_physical_cpus`
backs the `num_workers` default; `_looks_like_cnn` gates `cudnn.benchmark`.

Note `_looks_like_cnn` is duplicated in `auto_optim.py`. Two copies, one
concept — a reasonable consolidation target.

### `launcher.py` (153 LOC)

`_free_port`, `_rendezvous_env`, `_spawn_local_workers`, `_run_script_inplace`,
`launch`. This is what `autotrainer run` and the auto-spawn both go through.
`_rendezvous_env` is the multi-node correctness surface.

### `slurm.py` (205 LOC)

`is_slurm`, `node_scratch`, `apply` (exported as `configure_scratch`),
`configure_nccl`. `_looks_networked` is the heuristic that warns when your
"scratch" is actually NFS — the classic HPC footgun. `_detect_primary_interface`
picks the NCCL interface.

**Least validated module in the package.** Everything here is the SLURM path
that the 0/6 validation runbook has never exercised on real hardware.

### `metrics.py` (366 LOC — biggest blind spot)

**22 callables, 0 exported, 10 docstrings.** Nothing here is reachable from
`__all__` or pdoc, and it is the engine behind `metric=` (0.13).

Public-named API used by `tuning`/`fitting`: `resolve` (string → scorer),
`name_of`, `worst` (sentinel for "no score yet"), `is_better` (direction-aware
comparison), `score` (run a scorer over a loader).

Scorers are four small stateful classes with an `update`/`result` pair:
`_Accuracy`, `_F1`, `_Auc`, `_R2`. Streaming by design — they accumulate over
batches rather than materializing predictions. `_binary_auc` and `_average_ranks`
are the AUC implementation; `_grow_add` is the buffer growth strategy;
`_class_pred` / `_as_labels` normalize model output into labels.

If you touch one thing here, `is_better` is the one that silently corrupts a
search when wrong — a flipped comparison makes the tuner optimize backwards and
everything still looks like it ran fine.

### `sanity.py` (224 LOC)

The pre-flight data checks. `report` runs `_check_inputs` (normalization,
NaN/Inf, constant), `_check_targets`, `_check_classes` (imbalance). `overlap`
plus `_overlap_by_value` is train/val leak detection — index-based first, then
value-based. `_sample_rows` bounds the cost so checks stay cheap.

Warnings only; nothing is ever changed for the user.

### `preempt.py` (91 LOC)

`watch` arms a SIGUSR1 handler, `preempted()` is the flag the training loop
polls at epoch boundaries, `restore`/`reset` unwind it. Tiny module, but it is
the whole "survive a requeued SLURM job" story.

### `augment.py` (91 LOC)

One function, `augment_batch(x, strength)` — flip + cutout scaled by a single
searchable scalar. Deliberately not a policy engine: mixup/cutmix would have to
rewrite targets and the loss, which is a contract change, not a knob.

---

## Layer 4 — Observability

Three monitors, all opt-in, all zero-cost when not constructed, all
observe-only. Same shape: `tick`/`stats`/`should_report`/`report`.

- **`bottleneck.py`** (134) — `data_time` vs `step_time`. Is the loader
  starving the GPU?
- **`throughput.py`** (243) — samples/sec, peak memory, MFU.
  `_advertised_tflops` is a lookup table of device peak numbers; MFU is only
  computed when the caller supplies `model_flops`, because auto-counting FLOPs
  for an arbitrary model is unreliable enough that a wrong number beats no
  number in the worst way.
- **`triage.py`** (277) — `TrainingMonitor`. `_check_loss` (NaN/divergence),
  `_check_scaler` (fp16 overflow), `_check_grads` (non-finite/spike/vanish).
  `_warn` enforces fire-once-per-problem; `_grad_global_norm` is the hot path,
  called every step.

---

## Layer 5 — Orchestration

### `auto_optim.py` (505 LOC)

Inference of everything the user didn't specify. The chain:
`_peek_batch` → `_gather_targets` → `_infer_loss` → `_make_loss`, then
`_make_optimizer` / `_make_scheduler` / `_param_groups` / `_scale_lr`.

`_infer_loss` is the highest-leverage guess in the package — it decides
classification vs regression vs binary from a sampled batch, and everything
downstream (default search space, default metric) inherits that decision.
`_bce_loss` handles the binary special case.

`find_lr` / `_find_lr_synced` are the LR range test, on a throwaway model copy,
with the synced variant for distributed runs.

### `tuning.py` (448 LOC)

`_default_space` (recipe space chosen from model + inferred loss), `_suggest`
(Optuna trial → concrete config), `_rebuild_loader` (batch size is searchable,
so the loader is rebuilt per trial), `_evaluate` (score a trial).

### `tuning_estimator.py` (206 LOC)

The sklearn/XGBoost/LightGBM path. `_default_space` here is a curated per-family
table, not inferred. `_wants_n_jobs` and `_unpack` are adapter plumbing.

### `_fit_search.py` (153 LOC)

Phase-1 helpers carved out of `fitting.py`. **Zero public names.**
`_journal_storage` + `_parallel_search` are how trials get split across ranks
via a shared journal file — one trial per process. `_sync_from_rank0` and
`_unwrap` handle the distributed handoff; `_save_checkpoint` / `_load_checkpoint`
are the preemption-resume path.

### `fitting.py` (457 LOC, exactly one public function)

`fit()`. Phase 1 search, phase 2 retrain-the-winner-from-original-init. It
imports more than anything else in the package (8 internal modules) because it
is the composition point. NEXT_STEPS item 10 flags it for further splitting; it
has grown since that was written.

---

## Layer 6 — Surface

- **`__init__.py`** (216) — the dispatcher. 11 module-level functions, 9 public.
  Framework routing lives here, as do the `TypeError`s for removed kwargs.
- **`cli.py`** (51) — `run` / `info` / `doctor`. One function, **no docstring**.
- **`doctor.py`** (97) — `_check_frameworks`, `_check_gpu`, `_check_slurm`,
  `_check_port`, `run_doctor`. **Zero docstrings in the whole module.**
- **`__main__.py`** — `python -m autotrainer`.

---

## Where to start, by task

| Task | Start at |
|---|---|
| Change what `optimize=True` does | `_optimize.py`, then `torch_backend._loader_kwargs` |
| Add a selection metric | `metrics.resolve` + a new scorer class |
| Change the search space | `tuning._default_space`, `tuning._suggest` |
| Fix loss inference | `auto_optim._infer_loss` |
| Touch multi-node launch | `launcher._rendezvous_env`, `torch_backend._maybe_auto_launch` |
| Add a data check | `sanity.report` |
| Add a training-health check | `triage.TrainingMonitor._check_*` |

## Known soft spots

1. **`slurm.py` + multi-node launch** — the least validated code in the package;
   the runbook is 0/6.
2. **`metrics.py`** — 22 callables, nothing exported, no reference until this
   file. `is_better` fails silently when wrong.
3. **`fitting.py`** — 457 LOC and growing; already split once.
4. **`doctor.py` / `cli.py`** — zero docstrings between them.
5. **`_looks_like_cnn`** — duplicated in `_optimize.py` and `auto_optim.py`.
