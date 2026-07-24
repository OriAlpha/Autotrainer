"""Tests for ``prepare()``'s auto-launch: a fresh process on a multi-GPU box
spawns one worker per GPU and exits the parent, so a bare ``python train.py``
distributes without ``autotrainer run``.

The contract under test (``_maybe_auto_launch`` in torch_backend.py):

    Spawn + sys.exit ONLY when ALL three hold:
      1. no RANK / WORLD_SIZE env var (we're a fresh parent, not a worker)
      2. not under SLURM (srun already started the tasks)
      3. detect() reports local_multi_gpu (>= 2 GPUs on one box)

    In every other case: return silently, let prepare() proceed normally.

These tests never actually spawn real processes - subprocess.Popen is faked
so we assert on the env vars the parent would pass and on whether sys.exit
fires. The real end-to-end run lives in the cuda-marked CI on a multi-GPU
box.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from autotrainer.backends.torch_backend import _maybe_auto_launch  # noqa: E402
from autotrainer.detect import Environment  # noqa: E402


def _fake_worker_procs(n):
    """Return n MagicMock Popen objects that all report a clean exit (0)."""
    procs = []
    for _ in range(n):
        p = MagicMock()
        p.poll.return_value = 0
        p.pid = 1
        procs.append(p)
    return iter(procs)


class TestMaybeAutoLaunchNoSpawn:
    """The three conditions under which auto-launch must NOT fire. Each of
    these is a safety guard - getting any wrong means either an infinite
    spawn loop, a double-spawn under SLURM, or spawning on a single-GPU box.
    """

    def test_does_not_spawn_when_rank_set(self, monkeypatch):
        """Condition 1: a worker (launched by autotrainer run, srun, or a
        prior auto-launch) has RANK set and must NOT re-spawn - that would
        loop forever."""
        monkeypatch.setenv("RANK", "0")
        # detect() would say multi-GPU, and Popen would spawn - but neither
        # should even be called because the RANK check short-circuits first.
        spawn_called = MagicMock()
        monkeypatch.setattr("autotrainer.launcher._spawn_local_workers", spawn_called)
        _maybe_auto_launch()
        spawn_called.assert_not_called()

    def test_does_not_spawn_when_world_size_set(self, monkeypatch):
        """WORLD_SIZE set is equally a worker marker (some launchers set it
        before RANK). Treat either as 'already a worker'."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        spawn_called = MagicMock()
        monkeypatch.setattr("autotrainer.launcher._spawn_local_workers", spawn_called)
        _maybe_auto_launch()
        spawn_called.assert_not_called()

    def test_does_not_spawn_under_slurm(self, monkeypatch):
        """Condition 2: under SLURM, srun already started one task per GPU.
        Self-spawning here would double-spawn. The SLURM front door stays
        ``srun autotrainer run train.py``."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        spawn_called = MagicMock()
        monkeypatch.setattr("autotrainer.launcher._spawn_local_workers", spawn_called)
        _maybe_auto_launch()
        spawn_called.assert_not_called()

    def test_does_not_spawn_on_single_gpu(self, monkeypatch):
        """Condition 3: single-GPU (or CPU) boxes don't need spawning - the
        normal single-device prepare() path handles them."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setattr(
            "autotrainer.detect.detect",
            lambda: Environment(mode="single", gpus=1),
        )
        spawn_called = MagicMock()
        monkeypatch.setattr("autotrainer.launcher._spawn_local_workers", spawn_called)
        _maybe_auto_launch()
        spawn_called.assert_not_called()

    def test_does_not_spawn_on_cpu_zero_gpus(self, monkeypatch):
        """A CPU box (0 GPUs) must never try to spawn - detect() returns
        single with gpus=0, which is the no-GPU case."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setattr(
            "autotrainer.detect.detect",
            lambda: Environment(mode="single", gpus=0),
        )
        spawn_called = MagicMock()
        monkeypatch.setattr("autotrainer.launcher._spawn_local_workers", spawn_called)
        _maybe_auto_launch()
        spawn_called.assert_not_called()


class TestMaybeAutoLaunchSpawns:
    """When all three conditions hold, auto-launch spawns N workers (one per
    GPU) and the parent exits. We assert on the spawn call, the per-child env,
    and the sys.exit."""

    def _setup_multi_gpu_parent(self, monkeypatch, nproc=4):
        """Common setup: fresh parent (no RANK/SLURM), detect says
        local_multi_gpu, fake Popen returns clean workers."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(
            "autotrainer.detect.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=nproc, gpus=nproc),
        )
        # _spawn_local_workers calls subprocess.Popen and time.sleep; stub
        # both so no real process spawns and the poll loop doesn't block.
        procs = _fake_worker_procs(nproc)
        monkeypatch.setattr("autotrainer.launcher.subprocess.Popen", lambda *a, **kw: next(procs))
        monkeypatch.setattr("autotrainer.launcher.time.sleep", lambda *_: None)

    def test_spawns_one_worker_per_gpu(self, monkeypatch):
        """4 GPUs -> 4 Popen calls, each re-executing sys.argv."""
        self._setup_multi_gpu_parent(monkeypatch, nproc=4)
        popen_calls = []
        procs = _fake_worker_procs(4)
        monkeypatch.setattr(
            "autotrainer.launcher.subprocess.Popen",
            lambda *a, **kw: popen_calls.append((a, kw)) or next(procs),
        )
        with pytest.raises(SystemExit) as exc_info:
            _maybe_auto_launch()
        # Parent exits with the aggregate code (all workers 0 -> exit 0).
        assert exc_info.value.code == 0
        assert len(popen_calls) == 4

    def test_each_child_pinned_to_own_gpu(self, monkeypatch):
        """The spawned children get per-child CUDA_VISIBLE_DEVICES so each
        sees exactly one GPU (torchrun pattern - not inherited from parent)."""
        self._setup_multi_gpu_parent(monkeypatch, nproc=3)
        child_cvds = []
        procs = _fake_worker_procs(3)
        monkeypatch.setattr(
            "autotrainer.launcher.subprocess.Popen",
            lambda *a, env=None, **kw: (
                child_cvds.append(env["CUDA_VISIBLE_DEVICES"]) or next(procs)
            ),
        )
        with pytest.raises(SystemExit):
            _maybe_auto_launch()
        assert child_cvds == ["0", "1", "2"]

    def test_parent_exits_with_failing_code_on_worker_error(self, monkeypatch):
        """Fail-fast: if a worker exits non-zero, the parent's exit code
        propagates it (not 0) and the surviving worker is terminated."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(
            "autotrainer.detect.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=2, gpus=2),
        )
        dead = MagicMock()
        dead.poll.return_value = 7  # worker crashed
        dead.pid = 111
        survivor = MagicMock()
        survivor.poll.return_value = None
        survivor.pid = 222
        # Persistent iterator so Popen returns dead first, survivor second.
        proc_iter = iter([dead, survivor])
        monkeypatch.setattr(
            "autotrainer.launcher.subprocess.Popen", lambda *a, **kw: next(proc_iter)
        )
        monkeypatch.setattr("autotrainer.launcher.time.sleep", lambda *_: None)
        with pytest.raises(SystemExit) as exc_info:
            _maybe_auto_launch()
        assert exc_info.value.code == 7
        survivor.terminate.assert_called()

    def test_re_executes_sys_argv(self, monkeypatch):
        """The children re-run the SAME script the parent is running
        (sys.argv[0]) with the same args (sys.argv[1:]) - not a fixed name."""
        self._setup_multi_gpu_parent(monkeypatch, nproc=2)
        captured_cmds = []
        procs = _fake_worker_procs(2)
        monkeypatch.setattr(
            "autotrainer.launcher.subprocess.Popen",
            lambda *a, **kw: captured_cmds.append(a[0]) or next(procs),
        )
        monkeypatch.setattr("sys.argv", ["my_train_script.py", "--epochs", "10"])
        with pytest.raises(SystemExit):
            _maybe_auto_launch()
        # Each child's command starts with the python interpreter + the script.
        for cmd in captured_cmds:
            assert cmd[0].endswith(("python", "python.exe", "python3")) or "python" in cmd[0]
            assert cmd[1] == "my_train_script.py"
            assert "--epochs" in cmd and "10" in cmd


class TestAutoLaunchOptOut:
    """``prepare(..., auto_launch=False)`` must skip the spawn entirely - the
    escape hatch for users managing their own process spawning."""

    def test_auto_launch_false_skips_spawn(self, monkeypatch):
        """Even on a multi-GPU fresh parent, auto_launch=False means no spawn
        and prepare() proceeds as the single-process normal path."""
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Conditions that WOULD trigger a spawn if auto_launch were True.
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setattr(
            "autotrainer.detect.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=4, gpus=4),
        )
        spawn_called = MagicMock()
        monkeypatch.setattr("autotrainer.launcher._spawn_local_workers", spawn_called)

        from autotrainer.backends.torch_backend import prepare

        model = nn.Linear(3, 1)
        loader = DataLoader(TensorDataset(torch.randn(8, 3), torch.randn(8, 1)), batch_size=4)
        # Must NOT raise SystemExit - it returns normally.
        prepare(model, loader, auto_launch=False)
        spawn_called.assert_not_called()
