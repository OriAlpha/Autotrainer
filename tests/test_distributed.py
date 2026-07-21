"""Real multi-process distributed tests (2-rank gloo on CPU).

These spawn actual worker processes through the same rendezvous env vars the
launcher sets, then assert the cross-rank properties that single-process
unit tests cannot see: sampler sharding, LR-broadcast parity, and fit()
weight parity. Each worker deliberately seeds its loader shuffle with its
rank - the exact condition that used to desynchronize the ranks.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest

pytest.importorskip("torch")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_two_ranks(script: str, timeout: int = 240, extra_env: dict | None = None) -> list[str]:
    """Run `script` in 2 worker processes; return each rank's RESULT line.

    These tests validate the sharding / broadcast *logic* on CPU-gloo. They
    force CUDA off in the worker env so a 1-GPU dev box doesn't get tangled
    in CUDA init (two ranks can't both bind the same single GPU). Add a
    dedicated NCCL job for true multi-GPU validation.
    """
    port = _free_port()
    procs = []
    for rank in range(2):
        env = os.environ.copy()
        env.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            USE_LIBUV="0",  # Windows gloo needs the non-libuv TCP store
            CUDA_VISIBLE_DEVICES="",  # CPU-gloo: don't touch the GPU
        )
        env.update(extra_env or {})
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    results = []
    for rank, p in enumerate(procs):
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
                # Drain whatever the workers printed before we fail, so the
                # failure message shows *why* they hung, not just that they did.
            traces = []
            for r, q in enumerate(procs):
                try:
                    qo, qe = q.communicate(timeout=5)
                    traces.append(f"--- rank {r} stdout ---\n{qo}\n--- rank {r} stderr ---\n{qe}")
                except Exception:
                    traces.append(f"--- rank {r}: no output ---")
            pytest.fail(
                f"rank {rank} timed out after {timeout}s - likely a hang/desync\n"
                + "\n".join(traces)
            )
        assert p.returncode == 0, f"rank {rank} failed:\nstdout:\n{out}\nstderr:\n{err}"
        lines = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
        assert lines, f"rank {rank} printed no RESULT line:\n{out}"
        results.append(lines[-1])
    return results


_AUTO_WORKER = """
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import autotrainer

rank = int(os.environ["RANK"])
torch.manual_seed(0)  # identical model init on both ranks
x = torch.randn(64, 3)
y = x.sum(dim=1, keepdim=True)
g = torch.Generator().manual_seed(rank)  # per-rank shuffle order
loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, generator=g)

model = nn.Linear(3, 1)
model, loader, opt, loss_fn, sched = autotrainer.auto(model, loader, epochs=1)

assert isinstance(loader.sampler, DistributedSampler), "sampler not swapped"
autotrainer.set_epoch(loader, 0)
xb, yb = next(iter(loader))
opt.zero_grad()
loss_fn(model(xb), yb).backward()
opt.step()
lr = opt.param_groups[0]["lr"]
print(f"RESULT rank={rank} lr={lr:.12e} shard={len(loader.sampler)}")
"""

_FIT_WORKER = """
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer

rank = int(os.environ["RANK"])
torch.manual_seed(0)
x = torch.randn(64, 3)
y = x.sum(dim=1, keepdim=True)
g = torch.Generator().manual_seed(rank)
train = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, generator=g)
val = DataLoader(TensorDataset(x, y), batch_size=8)

model = nn.Linear(3, 1)
out, params, study = autotrainer.fit(
    model, train, val,
    trials=4, epochs=2, epochs_per_trial=1,
    space={"lr": ("loguniform", 1e-3, 1e-1)},
    study_storage=os.environ["TEST_STUDY_STORAGE"],
    verbose=False,
)
assert (study is not None) == (rank == 0), "study must exist on rank 0 only"
if study is not None:
    # Both ranks' trials must land in the one shared study (2 each).
    assert len(study.trials) == 4, f"expected 4 shared trials, got {len(study.trials)}"
assert not isinstance(out, nn.parallel.DistributedDataParallel)
w = torch.cat([p.detach().flatten() for p in out.parameters()]).double()
print(f"RESULT rank={rank} wsum={w.sum().item():.12e} lr={params['lr']:.12e}")
"""


class TestTwoRankGloo:
    def test_auto_shards_loader_and_broadcasts_lr(self):
        r0, r1 = _run_two_ranks(_AUTO_WORKER)
        # Both ranks must end up with the identical (rank-0) learning rate
        # despite differently-shuffled loaders, and 32 samples each.
        assert "shard=32" in r0 and "shard=32" in r1
        lr0 = r0.split("lr=")[1].split()[0]
        lr1 = r1.split("lr=")[1].split()[0]
        assert lr0 == lr1

    def test_fit_parallel_search_and_identical_weights_on_all_ranks(self, tmp_path):
        pytest.importorskip("optuna")
        storage = str(tmp_path / "study.log")
        r0, r1 = _run_two_ranks(_FIT_WORKER, extra_env={"TEST_STUDY_STORAGE": storage})
        # Same broadcast recipe + synced DDP training = bit-identical models.
        assert r0.split("rank=0 ")[1] == r1.split("rank=1 ")[1]
