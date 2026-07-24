"""PyTorch backend.

Called from user code via `autotrainer.prepare(...)`. Reads the env vars the
launcher set, initializes the process group if world_size > 1, wraps the
model in DDP, and swaps the DataLoader's sampler for a DistributedSampler.
"""

from __future__ import annotations

import os
from typing import Any


def _dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _ensure_process_group() -> bool:
    """Init the process group once if WORLD_SIZE > 1; True when distributed."""
    import torch
    import torch.distributed as dist

    rank, local_rank, world_size = _dist_info()
    if world_size <= 1:
        return False
    if not dist.is_initialized():
        # cuda_device() gates on device_count() > 0, not just is_available(),
        # so a driver-present but GPU-hidden box (e.g. CUDA_VISIBLE_DEVICES="")
        # falls through to gloo instead of crashing in set_device(N).
        from ..utils import cuda_device

        use_cuda = cuda_device(local_rank).type == "cuda"
        if use_cuda:
            # Bind before NCCL init so ranks don't all land on GPU 0.
            torch.cuda.set_device(local_rank)
        kwargs: dict[str, Any] = {}
        timeout_s = os.environ.get("AUTOTRAINER_TIMEOUT")
        if timeout_s:
            # Long rank-0 phases (e.g. fit()'s tuning) can outlive the
            # default collective timeout while the other ranks wait.
            from datetime import timedelta

            kwargs["timeout"] = timedelta(seconds=int(timeout_s))
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo",
            rank=rank,
            world_size=world_size,
            **kwargs,
        )
    return True


def _maybe_auto_launch() -> None:
    """If this is a fresh parent process on a multi-GPU box, spawn one worker
    per GPU and exit - so a bare ``python train.py`` distributes across all
    visible GPUs without the user invoking ``autotrainer run``.

    This is called at the very top of ``prepare()``. It spawns ONLY when all
    three hold:

    1. No ``RANK``/``WORLD_SIZE`` env var is set - we're a fresh parent, not a
       worker that the launcher (or a prior auto-launch) already spawned. The
       worker path has ``RANK`` set and must NOT re-spawn (infinite loop).
    2. Not under SLURM (``SLURM_JOB_ID`` absent). SLURM uses ``srun`` to start
       one task per GPU, so self-spawning here would double-spawn. The SLURM
       front door stays ``srun autotrainer run train.py``.
    3. ``detect()`` reports ``local_multi_gpu`` (>= 2 GPUs on this one box).

    When it fires, the parent process re-executes ``sys.argv`` once per GPU via
    :func:`autotrainer.launcher._spawn_local_workers` (each child pinned to its
    own GPU by ``CUDA_VISIBLE_DEVICES``), supervises them fail-fast, then
    ``sys.exit``\\ s with the aggregate exit code. The parent NEVER returns into
    the caller's training loop - that's the point. Each child re-imports,
    re-runs the user's script, and hits ``prepare()`` again, but now ``RANK`` is
    set so condition (1) fails and ``prepare()`` proceeds normally (DDP wrap,
    distributed sampler, etc.).

    This is opt-out: ``prepare(..., auto_launch=False)`` skips the call
    entirely (for users managing their own process spawning).
    """
    import sys

    # Condition 1: already a worker (launched by autotrainer run, srun, or a
    # prior auto-launch) - never re-spawn.
    if os.environ.get("RANK") is not None or os.environ.get("WORLD_SIZE") is not None:
        return
    # Condition 2: under SLURM, srun already started the tasks.
    if "SLURM_JOB_ID" in os.environ:
        return

    from ..detect import detect
    from ..launcher import _spawn_local_workers
    from ..utils import print0

    env = detect()
    # Condition 3: only self-spawn for local multi-GPU. Single-GPU and SLURM
    # modes are handled by the normal prepare() path / srun respectively.
    if env.mode != "local_multi_gpu":
        return

    print0(
        f"[autotrainer] auto-launch: {env.nproc_per_node} GPUs detected on one "
        f"node; spawning one worker per GPU (use prepare(..., auto_launch=False) "
        f"to disable, or `autotrainer run` for explicit control)"
    )
    rc = _spawn_local_workers(sys.argv[0], list(sys.argv[1:]), env)
    sys.exit(rc)


def _ddp_kwargs(
    local_rank: int, use_cuda: bool, static_graph: bool, find_unused_parameters: bool
) -> tuple[dict[str, Any], list[str]]:
    """Build ``DDP(...)`` constructor kwargs from the user's opts.

    Returns the kwargs plus the short tag names to record in ``applied`` for
    the summary line. ``static_graph=True`` also turns on
    ``gradient_as_bucket_view`` (peak-memory win that pairs naturally with
    a static graph); both are opt-in because they have correctness
    implications when the computation graph genuinely changes across iters.
    """
    kwargs: dict[str, Any] = {"device_ids": [local_rank] if use_cuda else None}
    tags: list[str] = []
    if static_graph:
        kwargs["static_graph"] = True
        # Bucket grads as views into the allreduce buffers: lower peak memory.
        kwargs["gradient_as_bucket_view"] = True
        tags.append("static_graph")
    if find_unused_parameters:
        kwargs["find_unused_parameters"] = True
        tags.append("find_unused_parameters")
    return kwargs, tags


def _loader_kwargs(dataloader: Any) -> dict[str, Any]:
    """Carry the user's DataLoader settings over to a rebuilt loader."""
    kwargs: dict[str, Any] = {
        "num_workers": dataloader.num_workers,
        "collate_fn": dataloader.collate_fn,
        "pin_memory": dataloader.pin_memory,
        "drop_last": dataloader.drop_last,
        "timeout": dataloader.timeout,
        "worker_init_fn": dataloader.worker_init_fn,
        "generator": dataloader.generator,
        "persistent_workers": dataloader.persistent_workers,
    }
    if dataloader.num_workers > 0:
        # DataLoader rejects prefetch_factor when there are no workers.
        kwargs["prefetch_factor"] = dataloader.prefetch_factor
    return kwargs


def _shard_loader(dataloader: Any, rank: int, world_size: int) -> Any:
    """Swap the loader's sampler for a DistributedSampler, keeping its settings."""
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    from torch.utils.data.distributed import DistributedSampler

    if isinstance(dataloader.dataset, torch.utils.data.IterableDataset):
        raise TypeError(
            "autotrainer cannot shard a DataLoader over an IterableDataset; "
            "shard inside the dataset (e.g. by RANK/WORLD_SIZE env vars) and "
            "it will be passed through unchanged."
        )
    if isinstance(dataloader.sampler, DistributedSampler):
        return dataloader  # the user already sharded it
    if dataloader.batch_size is None:
        raise TypeError(
            "autotrainer cannot shard a DataLoader built with batch_sampler=; "
            "construct it with batch_size= and let autotrainer install the "
            "DistributedSampler, or pass sampler=DistributedSampler(...) yourself."
        )
    # Honor the user's shuffle choice: SequentialSampler means shuffle=False.
    shuffle = not isinstance(dataloader.sampler, SequentialSampler)
    sampler = DistributedSampler(
        dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )
    if rank == 0:
        print(
            "[autotrainer] DistributedSampler installed (shuffle="
            f"{shuffle}) - call autotrainer.set_epoch(loader, epoch) at the "
            "start of every epoch so each epoch reshuffles"
        )
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        **_loader_kwargs(dataloader),
    )


def prepare(
    model: Any,
    dataloader: Any = None,
    optimizer: Any = None,
    *,
    optimize: bool = True,
    amp: bool | None = None,
    auto_bs: bool = False,
    loss_fn: Any = None,
    max_bs: int = 4096,
    compile: bool = False,
    compile_mode: str = "default",
    fsdp: bool = False,
    cpu_offload: bool = False,
    static_graph: bool = False,
    find_unused_parameters: bool = False,
    auto_launch: bool = True,
) -> Any:
    """Make (model, dataloader, optimizer) distribution-ready.

    Single device: returns inputs unchanged (moved to GPU if available).
    Multi device:  init process group, DDP-wrap model, distributed sampler.

    Args:
        model: a ``torch.nn.Module``.
        dataloader: optional ``DataLoader``.
        optimizer: optional optimizer; passed through untouched.
        optimize: turn on the GPU wins users forget (TF32, cudnn.benchmark
            for CNNs, ``num_workers``/``pin_memory``/``persistent_workers``
            defaults on bare loaders). **Default ``True``** - this is the
            no-brainer bundle; pass ``optimize=False`` to opt out. Never
            touches lr, loss, schedule, or optimizer choice. A no-op on CPU
            (every flag in it gates on a visible CUDA device), so CPU-only
            callers are unaffected by the default. The caller still wraps the
            forward/backward in ``autotrainer.autocast_context()`` /
            ``autotrainer.GradScaler()`` for AMP (both no-ops on CPU / when
            bf16 is supported); see the printed snippet when this fires.
        amp: enable mixed precision (bf16-preferred, fp16 fallback). Defaults
            to the value of ``optimize``. The caller wraps the forward in
            ``autotrainer.autocast_context()`` and the backward/step in
            ``autotrainer.GradScaler()`` (both no-ops on CPU / when bf16 is
            supported); see the printed snippet when this fires.
        auto_bs: grow the loader's batch size until OOM, then back off one
            step. Requires ``loss_fn`` for an accurate forward+backward
            measurement; without it the sweep is forward-only (conservative
            - it underestimates, since real training also needs memory for
            grads + optimizer state). The discovered batch size replaces
            the loader's; lr and schedule are NOT changed - raise your
            effective batch with :func:`autotrainer.accumulate` instead.
        loss_fn: the loss for the ``auto_bs`` forward+backward sweep. Not
            used for anything else; nothing is inferred or overridden.
        max_bs: ceiling for the ``auto_bs`` sweep (default 4096).
        compile: wrap the model with ``torch.compile()`` before any DDP wrap.
            Order matters: compiling the *unwrapped* module then DDP-wrapping
            is the documented-supported path; the reverse causes graph
            breaks on the ``.module`` indirection. No-op on CPU and on
            torch < 2.0 (which lacks ``torch.compile``). If compilation
            fails (e.g. dynamic shapes the backend can't handle), the
            uncompiled model is returned with a warning rather than crashing
            the run - check the log if you don't see the speedup.
        compile_mode: ``torch.compile`` mode (``default``, ``reduce-overhead``,
            ``max-autotune``). ``reduce-overhead`` uses CUDA graphs (fastest
            for small models with static shapes); ``max-autotune`` also
            searches kernel selections (slow first compile, best throughput).
        fsdp: wrap with ``FullyShardedDataParallel`` instead of ``DDP`` when
            distributed. FSDP shards parameters/grads/optimizer state across
            ranks, so a model too large to fit on one GPU still trains -
            DDP replicates and would OOM. Mutually exclusive with the plain
            DDP path. On single-device (world_size == 1) or torch < 2.0,
            ``fsdp=True`` is a no-op with a warning (FSDP only makes sense
            with >1 rank). Does NOT touch lr / loss / schedule / optimizer.
        cpu_offload: move FSDP parameters to CPU and bring them to GPU only
            for the forward/backward pass (``CPUOffload(offload_params=True)``).
            Trades compute throughput for the ability to train models that
            don't fit in GPU memory even when sharded across ranks. Only
            effective with ``fsdp=True``; ignored (with a warning) on the DDP
            path or single-process. Does NOT touch lr / loss / schedule.
        static_graph: when distributed (DDP path), enable DDP's
            ``static_graph=True`` plus ``gradient_as_bucket_view=True``.
            Both are free wins when the computation graph is the same every
            iteration (the common case): static graph skips per-iteration
            graph-recording overhead after the first step, and
            gradient-as-bucketing-view lowers peak memory by bucketing grads
            into the allreduce buffers. They are *opt-in* because they have
            correctness implications when the graph genuinely changes across
            iterations (conditional execution, varying depth) - enabling them
            silently there would hang or produce wrong grads. No-op on
            single-device and on the FSDP path. Does NOT touch lr / loss /
            schedule / optimizer. Mutually exclusive with
            ``find_unused_parameters`` (torch forbids the combination).
        find_unused_parameters: forward DDP's ``find_unused_parameters=True``
            to the DDP wrap. Needed when the model doesn't use all params every
            step (e.g. conditional branches) - without it DDP would hang. Only
            applies to the DDP path; no-op on single-device and FSDP.
        auto_launch: if ``True`` (default) and this is a fresh process on a
            multi-GPU box (no ``RANK`` set, not under SLURM, ``detect()``
            reports ``local_multi_gpu``), spawn one worker per GPU and exit
            the parent - so a bare ``python train.py`` distributes across all
            GPUs without ``autotrainer run``. Each worker re-enters this script
            with ``RANK`` set and proceeds normally. Set ``False`` to manage
            process spawning yourself (e.g. you already called ``autotrainer
            run`` or use your own launcher).
    """
    if auto_launch:
        _maybe_auto_launch()

    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    from .._optimize import apply_gpu_flags, build_loader_defaults, summarize

    rank, local_rank, world_size = _dist_info()
    # static_graph and find_unused_parameters are mutually exclusive - torch
    # raises an opaque error if both hit the DDP constructor. Catch it here
    # with a clear message instead. (Only matters on the DDP path, but this
    # is the natural place to validate since both are user-facing opts.)
    if static_graph and find_unused_parameters:
        raise ValueError(
            "prepare(static_graph=True, find_unused_parameters=True): "
            "torch DDP forbids this combination - a static graph can't also "
            "have unused parameters. Drop one."
        )
    # cuda_device() gates on device_count() > 0 so a driver-present but
    # GPU-hidden box doesn't try to bind a phantom device.
    from ..utils import cuda_device

    device = cuda_device(local_rank)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.set_device(device)
    model = model.to(device)

    # torch.compile: MUST happen before the DDP wrap. Compiling a
    # DDP-wrapped module is a documented footgun (graph breaks on the
    # .module indirection); the supported order is compile-then-wrap. On a
    # compile failure (dynamic shapes, unsupported op, ...) fall back to
    # the uncompiled model with a warning rather than killing the run.
    applied: dict[str, Any] = {}
    if compile and use_cuda and hasattr(torch, "compile"):
        from ..utils import print0

        try:
            model = torch.compile(model, mode=compile_mode)
            applied["compile"] = compile_mode
        except Exception as e:  # noqa: BLE001 - compile failures are varied
            print0(
                f"[autotrainer] compile: torch.compile failed ({type(e).__name__}: {e}); "
                "continuing with the uncompiled model"
            )
    elif compile and not hasattr(torch, "compile"):
        from ..utils import print0

        print0("[autotrainer] compile: torch.compile unavailable (torch < 2.0); skipping")

    # Optimize half: set the free-win flags before any work, so even the
    # DDP-wrap below benefits from TF32 etc. Track what we actually changed
    # so the summary is honest (apply_gpu_flags is a black box otherwise).
    if optimize and use_cuda:
        from .._optimize import _looks_like_cnn

        cnn = _looks_like_cnn(model)
        before = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.benchmark,
        )
        apply_gpu_flags(model, cnn=cnn)
        if torch.backends.cuda.matmul.allow_tf32 != before[0]:
            applied["tf32"] = True
        if cnn and torch.backends.cudnn.benchmark != before[1]:
            applied["cudnn_benchmark"] = True

    if amp is None:
        amp = optimize  # default: optimize= implies AMP, but stays overridable

    if world_size > 1:
        if dataloader is not None:
            # Shard (and validate) BEFORE any collective op: if this rank
            # raised after init, the others would hang in the process group.
            dataloader = _shard_loader(dataloader, rank, world_size)
        _ensure_process_group()
        if fsdp:
            # FSDP shards params/grads/optim state across ranks - the path
            # to take when the model is too large to replicate on every GPU
            # (where DDP OOMs). Falls back with a warning if FSDP isn't
            # available (torch < 2.0) - DDP is strictly better than crashing.
            from ..utils import print0

            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                fsdp_kwargs: dict[str, Any] = {
                    "device_id": local_rank if use_cuda else None,
                    "use_orig_params": True,  # so the user's optimizer still works
                }
                if cpu_offload:
                    # offload_params=True moves the sharded params to CPU and
                    # brings them to GPU only for fwd/bwd - the path when the
                    # model still OOMs even after sharding across ranks.
                    from torch.distributed.fsdp import CPUOffload

                    fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
                model = FSDP(model, **fsdp_kwargs)
                applied["wrap"] = "fsdp"
                if cpu_offload:
                    applied["cpu_offload"] = True
            except (ImportError, AttributeError):
                print0(
                    "[autotrainer] fsdp: FullyShardedDataParallel unavailable "
                    "(torch < 2.0); falling back to DDP"
                )
                ddp_kw, ddp_tags = _ddp_kwargs(
                    local_rank, use_cuda, static_graph, find_unused_parameters
                )
                model = DDP(model, **ddp_kw)
                applied["wrap"] = "ddp"
                if ddp_tags:
                    applied["ddp_opts"] = ddp_tags
                if cpu_offload:
                    print0(
                        "[autotrainer] cpu_offload: ignored (FSDP unavailable; "
                        "DDP has no built-in CPU param offload)"
                    )
        else:
            ddp_kw, ddp_tags = _ddp_kwargs(
                local_rank, use_cuda, static_graph, find_unused_parameters
            )
            model = DDP(model, **ddp_kw)
            applied.setdefault("wrap", "ddp")
            if ddp_tags:
                applied["ddp_opts"] = ddp_tags
            if cpu_offload:
                from ..utils import print0

                print0(
                    "[autotrainer] cpu_offload: ignored without fsdp=True "
                    "(DDP has no built-in CPU param offload; pair with fsdp=True)"
                )
    elif fsdp:
        # Single-process: FSDP buys nothing (no ranks to shard across) but
        # the user asked for it, so tell them rather than silently using DDP.
        from ..utils import print0

        print0("[autotrainer] fsdp: world_size == 1, FSDP is a no-op; model left unwrapped")
        if cpu_offload:
            print0("[autotrainer] cpu_offload: ignored (world_size == 1)")
    elif cpu_offload:
        # Single-process, no FSDP: cpu_offload is meaningless here too.
        from ..utils import print0

        print0(
            "[autotrainer] cpu_offload: ignored (world_size == 1 and fsdp=False; "
            "CPU param offload only applies to the FSDP path)"
        )
    if world_size <= 1 and (static_graph or find_unused_parameters):
        # static_graph / find_unused_parameters only affect the DDP wrap,
        # which never happens single-process. Tell the user rather than
        # silently dropping the opt-in.
        from ..utils import print0

        opts = []
        if static_graph:
            opts.append("static_graph")
        if find_unused_parameters:
            opts.append("find_unused_parameters")
        print0(
            f"[autotrainer] {'/'.join(opts)}: ignored (world_size == 1; "
            "these only apply to the multi-rank DDP path)"
        )

    # Loader optimizations run AFTER sharding so they see the final loader.
    # Only adds keys the user didn't set; merges with whatever _shard_loader
    # already preserved.
    if optimize and dataloader is not None and use_cuda:
        extra = build_loader_defaults(dataloader, world_size)
        if extra:
            from torch.utils.data import DataLoader

            merged = _loader_kwargs(dataloader)
            merged.update(extra)
            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=dataloader.sampler,
                **merged,
            )
            applied.update(extra)

    # Auto batch size: sweep after sharding + loader-kwargs so the test
    # runs against the same loader the user will train with. The sweep
    # rebuilds the loader once with the discovered size.
    if auto_bs and dataloader is not None and use_cuda:
        from ..utils import print0, robust_forward, split_xy, to_device

        # Without a loss_fn we can only measure activations+params, not grads
        # +optimizer state, so the picked batch size is safe but smaller than
        # what a real fwd+bwd sweep would allow. Surface this once so a user
        # surprised by a small size knows why (and how to get a bigger one).
        if loss_fn is None:
            print0(
                "[autotrainer] optimize: auto_bs running forward-only "
                "(no loss_fn); pass loss_fn for a larger batch size"
            )

        def sample_batch_fn(bs: int) -> None:
            from torch.utils.data import DataLoader as _DL

            tmp = _DL(dataloader.dataset, batch_size=bs, **_loader_kwargs(dataloader))
            xb, yb = split_xy(next(iter(tmp)))
            xb_dev = to_device(xb, device)
            opt_zero = torch.optim.SGD(model.parameters(), lr=1e-6)  # throwaway
            opt_zero.zero_grad(set_to_none=True)
            out = robust_forward(model, xb_dev)
            if loss_fn is not None:
                yb_dev = to_device(yb, device)
                loss_fn(out, yb_dev).backward()
                opt_zero.step()
            else:
                # Forward-only sweep: sum outputs to keep the graph, backprop
                # a scalar. Underestimates the safe batch (no real grads +
                # optimizer state) but never overestimates it.
                out.sum().backward()
                opt_zero.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        old_bs = dataloader.batch_size
        new_bs = find_batch_size(model, sample_batch_fn, start=max(2, old_bs), max_bs=max_bs)
        if new_bs != old_bs:
            from torch.utils.data import DataLoader

            merged = _loader_kwargs(dataloader)
            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=new_bs,
                sampler=dataloader.sampler,
                **merged,
            )
            applied["batch_size"] = (old_bs, new_bs)
            print0(
                f"[autotrainer] optimize: batch_size {old_bs} -> {new_bs} "
                f"(lr and schedule unchanged; use accumulate() to scale the step)"
            )

    # Summarize whenever the user opted into anything: optimize, compile, or
    # fsdp. The summarize() helper also gates internally, but checking here
    # keeps the call site readable and the intent explicit.
    if optimize or compile or fsdp:
        summarize(
            optimize=optimize,
            amp=bool(amp and use_cuda),
            applied=applied,
            compile=compile,
            fsdp=fsdp,
        )

    # When the optimize bundle fired on a GPU, the flags are applied but AMP
    # (autocast + GradScaler) still lives in the user's training loop - we
    # can't wrap an arbitrary loop from here. Print the exact two lines so
    # the user knows what to add; the helpers are no-ops on CPU so the
    # snippet is safe to copy verbatim. (fit() already does this internally
    # in its own loop, so this only matters for the manual-loop path.)
    if optimize and use_cuda and amp:
        from ..utils import print0

        print0(
            "[autotrainer] optimize: for AMP, wrap your step:\n"
            "    scaler = autotrainer.GradScaler()\n"
            "    with autotrainer.autocast_context():\n"
            "        out = model(x); loss = loss_fn(out, y)\n"
            "    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()"
        )

    out = [model]
    if dataloader is not None:
        out.append(dataloader)
    if optimizer is not None:
        out.append(optimizer)
    return out[0] if len(out) == 1 else tuple(out)


def find_batch_size(model: Any, sample_batch_fn: Any, start: int = 2, max_bs: int = 4096) -> int:
    """Double batch size until OOM, then back off one step.

    sample_batch_fn(bs) must run one forward+backward pass at batch size bs.
    """
    import torch

    bs = start
    best = start
    while bs <= max_bs:
        try:
            sample_batch_fn(bs)
            best = bs
            bs *= 2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            raise
    return best
