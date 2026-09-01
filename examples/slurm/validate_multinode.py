"""Multi-node SLURM validation: does autotrainer's DDP actually work across nodes?

Every check prints one PASS/FAIL line so the job output can be read top to
bottom. Nothing here is a benchmark - it asks only whether the distribution is
real: that the ranks landed on different physical nodes, that collectives cross
between them, that DDP genuinely synchronizes gradients, and that the sampler
shards the data instead of handing every rank the same rows.

Run it the documented way, from an sbatch script:

    srun autotrainer run examples/slurm/validate_multinode.py

The companion validate_multinode.sbatch does that and captures the environment
around it. A non-zero exit means at least one check failed.
"""

from __future__ import annotations

import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one check. Only rank 0 prints, so the log stays readable."""
    if not ok:
        FAILURES.append(name)
    if _rank() == 0:
        print(f"[CHECK] {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""),
              flush=True)
    return ok


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _gather(obj):
    """all_gather_object, or a one-element list when not distributed."""
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out: list = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def main() -> int:
    rank = _rank()
    host = socket.gethostname()

    # Every rank announces itself before any collective. If the job hangs after
    # this, the rendezvous is the problem, and this tells you which ranks got
    # far enough to try.
    print(f"[rank {rank}] host={host} "
          f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
          f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
          f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
          f"MASTER_PORT={os.environ.get('MASTER_PORT')} "
          f"SLURM_PROCID={os.environ.get('SLURM_PROCID')} "
          f"SLURM_NODEID={os.environ.get('SLURM_NODEID')} "
          f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))
    X = torch.randn(512, 16)
    y = torch.randint(0, 4, (512,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    # The call under test. Everything below asks whether it did its job.
    model, loader, opt = autotrainer.prepare(model, loader, opt)

    if rank == 0:
        print("\n" + "=" * 70 + "\nAUTOTRAINER MULTI-NODE VALIDATION\n" + "=" * 70, flush=True)

    distributed = dist.is_available() and dist.is_initialized()
    check("process group initialized", distributed,
          f"backend={dist.get_backend() if distributed else 'none'}")
    if not distributed:
        print("\nNot distributed - nothing below can be validated. Check that this ran "
              "under `srun` with --ntasks > 1.", flush=True)
        return 1

    world = dist.get_world_size()

    # --- 1. Did we really span nodes? -----------------------------------
    # The check the whole exercise exists for: if every rank reports the same
    # hostname, this was a single-node run wearing a multi-node job script and
    # proves nothing about cross-node rendezvous.
    hosts = _gather(host)
    nodes = sorted(set(hosts))
    check("ranks span >1 physical node", len(nodes) > 1,
          f"{len(nodes)} node(s): {nodes}")

    expected = int(os.environ.get("SLURM_NTASKS", world))
    check("world_size matches SLURM_NTASKS", world == expected,
          f"world_size={world} SLURM_NTASKS={expected}")

    per_node = {n: hosts.count(n) for n in nodes}
    check("tasks spread evenly over nodes", len(set(per_node.values())) == 1,
          f"{per_node}")

    # --- 2. One distinct GPU per rank -----------------------------------
    # The device comes from the model prepare() already placed, so this can
    # never disagree with where training actually runs. Gated on device_count,
    # not is_available: with CUDA_VISIBLE_DEVICES="" a driver-present box
    # reports available=True and then has no device 0 to put a tensor on.
    device = next(model.parameters()).device
    if device.type == "cuda":
        pairs = _gather((host, device.index))
        check("each rank owns a distinct GPU", len(set(pairs)) == world, f"{sorted(pairs)}")
    else:
        # Not a failure by itself - on a CPU allocation this is correct. It is
        # only wrong if you asked for --gres=gpu, which section 3 of the sbatch
        # output will show you did.
        print(f"[note] model is on {device}, not a GPU - per-GPU checks skipped. "
              f"Expected only if this job requested no --gres=gpu.", flush=True)

    # --- 3. Do collectives actually cross the interconnect? -------------
    # Sum of ranks has one right answer. A wrong sum means the ranks are not in
    # the group they think they are; a hang here means the rendezvous is broken.
    t = torch.tensor([float(rank)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    want = world * (world - 1) / 2
    check("all_reduce spans the whole group", abs(t.item() - want) < 1e-6,
          f"got {t.item():.0f}, expected {want:.0f}")

    # --- 4. Does the sampler actually shard? ----------------------------
    # The failure this catches is every rank training on the identical rows,
    # which looks like it works and silently wastes every GPU but one.
    autotrainer.set_epoch(loader, 0)
    my_samples = sum(xb.shape[0] for xb, _ in loader)
    all_counts = _gather(my_samples)
    check("each rank gets an equal shard", len(set(all_counts)) == 1,
          f"samples/rank={all_counts} total={sum(all_counts)} dataset=512")
    check("shards cover the dataset", abs(sum(all_counts) - 512) <= world,
          f"{sum(all_counts)} vs 512")
    check("each rank sees less than the whole dataset", my_samples < 512 or world == 1,
          f"this rank saw {my_samples} of 512")

    # --- 5. Does DDP actually synchronize gradients? --------------------
    # The real proof. Each rank steps on its own shard; if gradients are being
    # all-reduced, every rank's weights stay bit-identical afterwards. If DDP
    # is a no-op wrapper, the replicas drift apart on the first step.
    loss_fn = nn.CrossEntropyLoss()
    autotrainer.set_epoch(loader, 0)
    model.train()
    first_loss = last_loss = None
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        if step >= 30:
            break

    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    checksum = float(flat.double().sum().item())
    sums = _gather(checksum)
    spread = max(sums) - min(sums)
    check("weights identical across ranks after stepping", spread < 1e-6,
          f"max-min={spread:.3e} over {world} ranks")

    losses = _gather(last_loss)
    check("every rank made progress", all(x is not None for x in losses),
          f"first={first_loss:.4f} last/rank={[round(x, 4) for x in losses]}")

    # --- 6. Rank-aware helpers ------------------------------------------
    check("rank()/is_main() agree with the env", autotrainer.rank() == rank
          and autotrainer.is_main() == (rank == 0),
          f"autotrainer.rank()={autotrainer.rank()} is_main={autotrainer.is_main()}")

    mains = _gather(autotrainer.is_main())
    check("exactly one rank is main", sum(1 for m in mains if m) == 1, f"{mains}")

    # --- 7. The CPU allocation --------------------------------------
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    workers = getattr(loader, "num_workers", None)
    if cpus:
        check("num_workers within the CPU allocation", workers is not None
              and workers <= int(cpus),
              f"num_workers={workers} SLURM_CPUS_PER_TASK={cpus}")
    else:
        print(f"[note] SLURM_CPUS_PER_TASK unset; num_workers={workers}", flush=True)

    # --- 8. Auto-spawn must stand down under SLURM ----------------------
    # If it did not, there would be more processes than SLURM tasks and the
    # world size would disagree with SLURM_NTASKS - already checked above, so
    # this just reports what the launcher decided.
    check("launcher reports slurm mode", os.environ.get("AUTOTRAINER_MODE") == "slurm",
          f"AUTOTRAINER_MODE={os.environ.get('AUTOTRAINER_MODE')}")

    dist.barrier()
    if rank == 0:
        print("=" * 70, flush=True)
        if FAILURES:
            print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED -> {FAILURES}", flush=True)
        else:
            print(f"RESULT: ALL CHECKS PASSED on {len(nodes)} node(s), {world} ranks", flush=True)
        print("=" * 70, flush=True)

    autotrainer.finish(cleanup_dist=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
