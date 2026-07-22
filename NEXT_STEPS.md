# Next steps

Concrete follow-ups from the GPU-optimization work landed in PRs #1–#5.
Each item carries the context needed to pick it up later without
re-deriving it — the why, the scope, and the gotchas that bit during
implementation. The README's Roadmap section is the product-level vision;
this doc is the engineering backlog that came out of this stretch of work.

Priority ordering is a suggestion, not a verdict — pick what's most
useful to you next.

---

## Tier 4: the remaining GPU-throughput knobs

Small, well-scoped wins that fit the same `prepare(optimize=True)` contract
(detect + optimize, leave hyperparameters alone). None touch lr / loss /
schedule / optimizer.

### 1. DDP `gradient_as_bucketing_view` + `static_graph`
**Effort:** small (~20 lines + test).
**Payoff:** free DDP speedups when the computation graph is static across
iterations (the common case). `gradient_as_bucketing_view=True` reduces
peak memory by bucketing grads; `static_graph=True` skips per-iteration
graph-recording overhead after the first step.
**Where:** `backends/torch_backend.py`, the DDP-wrap branch in `prepare()`.
Pass them as kwargs to the `DDP(...)` constructor, but only when the user
opts in (a new `prepare(..., static_graph=True)` flag) — both have
correctness implications if the graph genuinely changes (conditional
execution, varying depth), so they shouldn't be silently on.
**Test:** extend `test_tier3.py::TestFsdp` with a `TestDdpOpts` class;
CPU-gated (the DDP-wrap path needs `world_size > 1`, so use the existing
2-rank gloo distributed-test harness with `CUDA_VISIBLE_DEVICES=""`).
**Gotcha:** `static_graph=True` cannot be combined with `find_unused_parameters`;
if a user later passes that, raise a clear error rather than letting
torch emit an opaque one.

### 2. Throughput / MFU logging
**Effort:** medium (~80 lines).
**Payoff:** lets the user *see* they're optimized — answers "am I
GPU-bound or waiting on the loader?" without a profiler. Pairs naturally
with the existing `BottleneckMonitor`.
**Where:** a new `throughput.py` module, exposed as
`autotrainer.ThroughputMonitor` (or fold into `bottleneck.py`). Track
samples/sec, peak GPU memory, and a rough MFU estimate (achieved
FLOPs / advertised FLOPs at the detected GPU's spec). The MFU denominator
needs a small table of GPU specs (A100 312 TFLOPS, H100 989 TFLOPS bf16,
RTX 5070 ~? TFLOPS) — keep it tiny, or read from
`torch.cuda.get_device_properties()` where possible.
**Test:** unit test the bookkeeping (counter increments, time-window
averaging) on CPU; gate the real-GPU-memory read on the `cuda` marker.
**Gotcha:** MFU is approximate and model-dependent; present it as a
diagnostic, not a hard metric. Document that the FLOPs estimate assumes
the model is matmul-heavy (true for transformers/CNNs, not for RNNs or
sparse models).

### 3. NCCL env tuning for multi-node SLURM
**Effort:** medium (~60 lines + a real multi-node test, which is the hard part).
**Payoff:** multi-node SLURM jobs frequently hang or underperform because
NCCL can't infer the right network interface and falls back to a slow one.
**Where:** extend `slurm.py` (which already has `configure_scratch()`).
Add a `configure_nccl()` that detects the non-loopback interface
(`hostname -I`, parse `ip route`, or shell out to `ifconfig`) and sets
`NCCL_SOCKET_IFNAME` if it's unset. Optionally set `NCCL_DEBUG=INFO` on
the first run of a job so the user can see what NCCL chose.
**Gotcha:** this is genuinely hard to test without a real multi-node
cluster. The 1-rank RTX 5070 box can't exercise it. Gate the logic test
on monkeypatched env (like `test_tier3b.py` does for the scratch helper),
and leave the real validation as a 1.0-blocker item (see below).

---

## Testing gaps surfaced during this work

### 4. Multi-rank FSDP test (extend the distributed harness)
**Effort:** medium.
**Payoff:** closes the biggest "wired but unproven" gap in the Tier 3
work. Right now `prepare(fsdp=True)` is only exercised in the single-
process no-op path (`test_fsdp_single_process_is_noop_with_warning`). The
actual FSDP wrap + `use_orig_params=True` + `CPUOffload` path has never
run against a real process group.
**Where:** `tests/test_distributed.py` — extend the existing 2-rank gloo
harness (which already forces `CUDA_VISIBLE_DEVICES=""`) to spawn a worker
that calls `prepare(fsdp=True)` and asserts the returned model is an FSDP
instance with the right number of parameter groups. FSDP works on CPU-gloo
for testing purposes (it's the *sharding* that matters, not the transport).
**Gotcha:** FSDP on CPU-gloo is slower than DDP-gloo; keep the model tiny
(one `nn.Linear`) and the dataset to ~16 samples. If it turns out FSDP
needs CUDA even to shard (some torch versions did), gate the test on
`device_count > 0` + the `cuda` marker and accept that the 1-GPU box
can only test 1-rank FSDP, not 2-rank.

### 5. `torch.compile` end-to-end test on a Triton-capable runner
**Effort:** trivial once the runner is in place (it already is).
**Payoff:** `TestCompile::test_compile_then_backward_runs` is currently
skipped on the Windows runner because native Windows lacks Triton. The
test would run on a Linux GPU runner; until then the compile *wiring* is
tested but the compiled-graph forward+backward is not.
**Where:** no code change — the test already exists and is correctly
gated. The action is: if you ever register a Linux GPU runner, this test
will un-skip automatically. Track it as "known coverage gap" rather than
something to fix.

---

## Infrastructure debt

### 6. The `test-cuda` job is pinned to one machine
**Effort:** small (docs + maybe a fallback).
**Payoff:** resilience — if your RTX 5070 box is offline, `test-cuda`
shows as "queued" forever and GPU coverage silently drops to zero.
**Where:** `RUNNER_SETUP.md` already documents the one-machine assumption.
Options to harden: (a) add a job-level timeout so a dead runner shows as
"failed after 30min" instead of "queued" forever (visible signal); (b)
register a second runner (even a CPU-only `self-hosted` one, labeled
differently, just to keep the Actions plumbing warm); (c) accept the
single-point-of-failure for a personal project.
**Gotcha:** option (a) is the highest value per effort. A `timeout-minutes: 30`
on the `test-cuda` job turns "silently stuck" into "obviously failed",
which is much easier to notice.

### 7. The Python version on the runner is manually pinned
**Effort:** small.
**Payoff:** the `test-cuda` job's `env.PYTHON` is hardcoded to
`C:\Python313\python.exe`. When Python 3.13 goes EOL or you want to test
on 3.14, you have to remember to update both the runner box *and* the
ci.yml path. A `RUNNER_SETUP.md` note helps but doesn't prevent drift.
**Where:** consider a tiny `scripts/provision-runner-python.ps1` that
copies uv's Python to `C:\Python<ver>` and applies the ACLs — runnable
from the runner box, idempotent, and the single source of truth the
ci.yml path points at. Low priority; the current manual process works.

---

## The 1.0 gate

### 8. Real multi-node SLURM validation
**Effort:** large, and not really code — it's an integration-validation
exercise.
**Payoff:** this is the single biggest unknown in the project. The README
explicitly gates 1.0 on it. Every distributed-path test today is 2-rank
gloo on a single box; multi-node NCCL across separate hosts has never run.
**Where:** not a PR — it's a test campaign. Borrow a 2-node SLURM
allocation, run `examples/pytorch_ddp.py` and a `fit()` job end-to-end,
and document what breaks. Likely candidates for breakage: the
`_slurm_master_addr()` nodelist parser (line 83 of `detect.py`) on a
non-trivial hostlist; the `AUTOTRAINER_PORT` default of 29500 if two jobs
share a node; the Optuna journal-file storage in `fit()` over NFS
(`configure_scratch()` mitigates but hasn't been proven on a real NFS mount).
**Gotcha:** a single successful 2-node run is not validation — it's one
data point. The gate should be "ran N times across M days without a
distributed-path bug", which is hard to shortcut.

### 9. Deprecation removal: `train_loader=` / `val_loader=`
**Effort:** trivial (~10 lines + CHANGELOG).
**Payoff:** closes the 0.10 deprecation cycle, unblocks 1.0.
**Where:** `__init__.py` lines ~138–153 — delete the alias-shimming block
and the `DeprecationWarning`. The soak period has elapsed since 0.10;
the only question is whether to bundle this with the multi-node
validation (item 8) as part of a single 1.0 release, or ship it earlier
as 0.11.
**Gotcha:** check PyPI download stats / GitHub search for anyone still
passing the old kwargs before removing. Low traffic, but a one-line
removal is a real breaking change for anyone who didn't read the
deprecation warning.

---

## Smaller polish (low priority, grab-bag)

### 10. `fitting.py` is still 431 LOC doing too much
Phase-1 distributed Optuna search + phase-2 full retrain + checkpoint
versioning + recipe/loss broadcast + early stopping. The roadmap's
"training triage" features will compound that complexity. Worth splitting
along the phase boundary (`_search.py` + `_train.py`) before adding more.
**Effort:** medium refactor, no behavior change. Do it as a pure-move PR
first, then start layering features.

### 11. `auto_bs` forward-only sweep is conservative
Without `loss_fn` it only measures activations+params, not grads+optimizer
state — so it picks a batch size that's safe but smaller than what fwd+bwd
would allow. The docstring documents this, but a user who passes no
`loss_fn` and gets a surprisingly small batch size may be confused.
Consider printing a one-line note when the forward-only sweep is used:
"forward-only sweep; pass loss_fn for a larger batch size".

### 12. The `_pretend_cuda` test stub is brittle
It patches `is_available`, `device_count`, `set_to_device`, and `Tensor.to`
individually. If `prepare()` starts using another CUDA entry point, the
stub silently exercises the wrong path again (exactly the bug that bit in
PR #1's CI). Consider a single `conftest.py` fixture that monkeypatches a
broader surface or — better — mocks `torch.cuda` as a whole for the
optimize-path tests. Lower priority now that the real GPU CI catches
these, but still a code smell.
