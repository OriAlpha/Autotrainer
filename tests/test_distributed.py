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


# Exercises the DDP-opts path (prepare(static_graph=True)) over the same
# 2-rank gloo harness. static_graph + gradient_as_bucketing_view are the
# free DDP wins when the computation graph is static across iterations; both
# need a real process group (world_size > 1) so they can't be tested in
# single-process unit tests. The mutual-exclusion guard runs in-process.
_DDP_OPTS_WORKER = """
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer

rank = int(os.environ["RANK"])
torch.manual_seed(0)
x = torch.randn(32, 3)
y = x.sum(dim=1, keepdim=True)
loader = DataLoader(TensorDataset(x, y), batch_size=8)

model = nn.Linear(3, 1)
model, loader = autotrainer.prepare(model, loader, static_graph=True)
assert isinstance(model, nn.parallel.DistributedDataParallel), "model not DDP-wrapped"
assert model.static_graph is True, "static_graph did not reach the DDP constructor"
print(f"RESULT rank={rank} ok=1 wrap=ddp")
"""


class TestDdpOpts:
    """DDP constructor opts (static_graph, find_unused_parameters).

    NEXT_STEPS suggested living in test_tier3.py, but that module is
    cuda-gated (module-level skipif no GPU) while these run on CPU-gloo -
    so they belong here next to the other 2-rank gloo tests.
    """

    def test_static_graph_wired_into_ddp(self):
        r0, r1 = _run_two_ranks(_DDP_OPTS_WORKER)
        assert "wrap=ddp" in r0 and "wrap=ddp" in r1

    def test_static_graph_and_find_unused_parameters_are_mutually_exclusive(self):
        """torch forbids static_graph + find_unused_parameters; prepare must
        raise a clear ValueError rather than letting torch emit an opaque one
        deep inside DDP init."""
        import torch.nn as nn

        from autotrainer.backends.torch_backend import prepare

        model = nn.Linear(3, 1)
        with pytest.raises(ValueError, match="mutually exclusive|forbids this combination"):
            prepare(model, static_graph=True, find_unused_parameters=True)

    def test_static_graph_single_process_is_noop_with_warning(self, capsys, monkeypatch):
        """world_size == 1: no DDP wrap happens, so static_graph is a no-op.
        Must warn rather than silently dropping the opt-in."""
        import torch.nn as nn

        from autotrainer.backends.torch_backend import prepare

        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        model = nn.Linear(3, 1)
        out = prepare(model, static_graph=True)
        assert not isinstance(out, nn.parallel.DistributedDataParallel)
        assert "static_graph" in capsys.readouterr().out


# Closes the gap the single-process test (test_tier3.py::TestFsdp) leaves open:
# that one only checks the world_size==1 no-op. The multi-rank FSDP *wrap* had
# never run against a real process group in CI. The wrap + use_orig_params
# param-addressability check run fine on CPU-gloo and is what's exercised here.
#
# The full sharded fwd+bwd+step is a different story: on torch 2.13, FSDP
# refuses to run a forward when params are on CPU if `cuda.is_available()` is
# True ("An FSDP-managed module unexpectedly has parameters on cpu. Move the
# module to cuda:0 before training."). On this test box the CUDA driver is
# present but the GPU is hidden (CUDA_VISIBLE_DEVICES="" -> device_count=0),
# so FSDP insists on a cuda:0 that doesn't exist - the step can't run. That
# full-step path is therefore gated on a real, usable GPU (device_count > 0)
# and left to the cuda-marked CI. This is exactly the contingency NEXT_STEPS
# item #4 flagged. The wrap itself - the part that was genuinely unproven -
# runs here on CPU-gloo.
_FSDP_WRAP_WORKER = """
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer

rank = int(os.environ["RANK"])
torch.manual_seed(0)
x = torch.randn(16, 3)
y = x.sum(dim=1, keepdim=True)
loader = DataLoader(TensorDataset(x, y), batch_size=4)

model = nn.Linear(3, 1)
m, loader = autotrainer.prepare(model, loader, fsdp=True)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
assert isinstance(m, FSDP), "model not FSDP-wrapped"

# use_orig_params=True is what lets the user's optimizer keep working under
# FSDP - assert params are still addressable by name (not flattened away).
named = dict(m.named_parameters())
assert any("weight" in k for k in named), "weight not addressable (use_orig_params?)"
assert any("bias" in k for k in named), "bias not addressable (use_orig_params?)"
print(f"RESULT rank={rank} ok=1 wrap=fsdp nparams={len(named)}")
"""


class TestFsdpMultiRank:
    """The multi-rank FSDP wrap path (prepare(fsdp=True) over a real group).

    Lives here rather than test_tier3.py (as NEXT_STEPS suggested) because
    test_tier3.py is cuda-gated at module level, and the FSDP *wrap* +
    use_orig_params path runs fine on CPU-gloo - that's the part that was
    wired but never proven. The full sharded step needs a usable GPU (see
    the _FSDP_WRAP_WORKER comment) and is left to the cuda-marked CI.
    """

    def test_fsdp_wraps_with_orig_params_over_process_group(self):
        """The wrap itself + use_orig_params param addressability - the gap the
        single-process no-op test leaves open."""
        r0, r1 = _run_two_ranks(_FSDP_WRAP_WORKER)
        assert "wrap=fsdp" in r0 and "wrap=fsdp" in r1

    def test_fsdp_with_cpu_offload_wraps(self):
        """prepare(fsdp=True, cpu_offload=True) must wrap without crashing in
        the multi-rank path. (A full step isn't run - see the worker comment for
        why; this just proves the cpu_offload kwarg is plumbed through FSDP
        init across ranks.)"""
        worker = _FSDP_WRAP_WORKER.replace(
            "autotrainer.prepare(model, loader, fsdp=True)",
            "autotrainer.prepare(model, loader, fsdp=True, cpu_offload=True)",
        )
        r0, r1 = _run_two_ranks(worker)
        assert "wrap=fsdp" in r0 and "wrap=fsdp" in r1

    @pytest.mark.cuda
    def test_fsdp_runs_a_sharded_forward_backward_step(self):
        """Full sharded fwd+bwd+step. Needs at least 2 usable GPUs (one per
        rank): on torch 2.13 FSDP won't run a forward with params on CPU when
        cuda.is_available() is True, so each rank needs its own real device.
        Skipped on the single-GPU dev box and in CPU-only CI - this is real
        multi-GPU validation, left to a runner that has it."""
        import torch

        if torch.cuda.device_count() < 2:
            pytest.skip("FSDP sharded step needs >= 2 usable GPUs (one per rank)")

        worker = """
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer

rank = int(os.environ["RANK"])
torch.manual_seed(0)
x = torch.randn(16, 3)
y = x.sum(dim=1, keepdim=True)
loader = DataLoader(TensorDataset(x, y), batch_size=4)

model = nn.Linear(3, 1)
# CUDA is NOT hidden here - each rank gets its own device via LOCAL_RANK so
# FSDP can shard onto it and move inputs there.
m, loader = autotrainer.prepare(model, loader, fsdp=True)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
assert isinstance(m, FSDP), "model not FSDP-wrapped"

autotrainer.set_epoch(loader, 0)
xb, yb = next(iter(loader))
device = next(m.parameters()).device
xb, yb = xb.to(device), yb.to(device)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
opt.zero_grad()
loss = ((m(xb) - yb) ** 2).mean()
loss.backward()
opt.step()
assert torch.isfinite(loss), "loss is not finite after a sharded step"
print(f"RESULT rank={rank} ok=1 wrap=fsdp loss={loss.item():.6e}")
"""
        # Don't hide CUDA here - the workers need real devices. LOCAL_RANK in
        # the harness already pins each rank to its own GPU via CUDA_VISIBLE_DEVICES
        # ... but _run_two_ranks forces CUDA_VISIBLE_DEVICES="". Override that
        # by letting the workers see all devices; torch's default local-rank
        # binding + prepare()'s set_device handles the split.
        r0, r1 = _run_two_ranks(worker, extra_env={"CUDA_VISIBLE_DEVICES": "0,1"})
        assert "wrap=fsdp" in r0 and "wrap=fsdp" in r1
        for r in (r0, r1):
            assert "loss=" in r
            loss = float(r.split("loss=")[1].split()[0])
            assert loss == loss  # NaN check (NaN != NaN)
