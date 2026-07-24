"""Tests for the process launcher (launcher.py).

The launcher's three branches (single / local_multi_gpu / slurm) are hard to
exercise end-to-end without real GPUs or a cluster, so we drive them with
monkeypatched `detect()` and (for the multi-GPU path) fake subprocesses. The
rendezvous-env translation is pure and tested directly.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

from autotrainer.detect import Environment
from autotrainer.launcher import _rendezvous_env, launch


class TestRendezvousEnv:
    def test_sets_all_standard_vars(self):
        env = Environment(
            mode="local_multi_gpu", nnodes=2, nproc_per_node=2, master_addr="10.0.0.1"
        )
        e = _rendezvous_env(env, rank=2, local_rank=1)
        assert e["RANK"] == "2"
        assert e["LOCAL_RANK"] == "1"
        assert e["WORLD_SIZE"] == "4"  # nnodes(2) * nproc(2)
        assert e["MASTER_ADDR"] == "10.0.0.1"
        assert e["MASTER_PORT"] == "29500"
        assert e["AUTOTRAINER_ACTIVE"] == "1"
        assert e["AUTOTRAINER_MODE"] == "local_multi_gpu"

    def test_preserves_existing_env(self, monkeypatch):
        monkeypatch.setenv("PRE_EXISTING", "keep_me")
        env = Environment(mode="single")
        e = _rendezvous_env(env, rank=0, local_rank=0)
        assert e["PRE_EXISTING"] == "keep_me"


class TestLaunchSingle:
    def test_single_mode_sets_env_and_runs_script(self, tmp_path: Path, monkeypatch):
        # A script that writes the RANK it sees to a file we can read back.
        out_file = tmp_path / "rank.txt"
        script = tmp_path / "dump_rank.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import os
                with open(r"{out_file}", "w") as f:
                    f.write(os.environ.get("RANK", "MISSING"))
                """
            )
        )
        # Force single-device mode regardless of the test runner's GPU count.
        monkeypatch.setattr(
            "autotrainer.launcher.detect",
            lambda: Environment(mode="single", gpus=0),
        )
        rc = launch(str(script), [])
        assert rc == 0
        assert out_file.read_text() == "0"


class TestLaunchSlurm:
    def test_slurm_translates_procid_to_rank(self, tmp_path: Path, monkeypatch):
        out_file = tmp_path / "vars.txt"
        script = tmp_path / "dump_vars.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import os
                with open(r"{out_file}", "w") as f:
                    f.write(os.environ.get("RANK", "?") + "\\n")
                    f.write(os.environ.get("LOCAL_RANK", "?") + "\\n")
                    f.write(os.environ.get("WORLD_SIZE", "?") + "\\n")
                """
            )
        )
        monkeypatch.setenv("SLURM_PROCID", "3")
        monkeypatch.setenv("SLURM_LOCALID", "1")
        monkeypatch.setattr(
            "autotrainer.launcher.detect",
            lambda: Environment(
                mode="slurm", nnodes=2, nproc_per_node=2, node_rank=1, master_addr="n01"
            ),
        )
        rc = launch(str(script), [])
        assert rc == 0
        lines = out_file.read_text().splitlines()
        assert lines == ["3", "1", "4"]  # RANK, LOCAL_RANK, WORLD_SIZE(2*2)


class TestLaunchLocalMultiGpu:
    def test_kills_survivors_on_worker_failure(self, monkeypatch):
        """If one worker exits non-zero, the rest are terminated immediately."""
        # Two fake procs: the first exits non-zero, the second is still running
        # (poll() returns None) and must get .terminate() called on it.
        dead_worker = MagicMock()
        dead_worker.poll.return_value = 1  # exited with error
        dead_worker.pid = 111

        survivor = MagicMock()
        survivor.poll.return_value = None  # still running
        survivor.pid = 222

        # Popen returns the dead worker on the first call and the survivor on
        # the second. Use a persistent iterator so each call yields the next
        # proc (a list-literal lambda would re-create the list every call).
        proc_iter = iter([dead_worker, survivor])
        monkeypatch.setattr(
            "autotrainer.launcher.subprocess.Popen",
            lambda *a, **kw: next(proc_iter),
        )
        # Shorten the poll loop sleep so the test doesn't wait.
        monkeypatch.setattr("autotrainer.launcher.time.sleep", lambda *_: None)
        monkeypatch.setattr(
            "autotrainer.launcher.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=2, gpus=2),
        )

        rc = launch("dummy_script.py", [])
        assert rc == 1  # propagated from the dead worker
        survivor.terminate.assert_called()

    def test_clean_exit_returns_zero(self, monkeypatch):
        """All workers exiting 0 yields return code 0."""
        p0, p1 = MagicMock(), MagicMock()
        p0.poll.return_value = 0
        p0.pid = 1
        p1.poll.return_value = 0
        p1.pid = 2
        proc_iter = iter([p0, p1])
        monkeypatch.setattr(
            "autotrainer.launcher.subprocess.Popen", lambda *a, **kw: next(proc_iter)
        )
        monkeypatch.setattr("autotrainer.launcher.time.sleep", lambda *_: None)
        monkeypatch.setattr(
            "autotrainer.launcher.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=2, gpus=2),
        )
        assert launch("dummy_script.py", []) == 0

    def test_picks_free_port_not_the_fixed_default(self, monkeypatch):
        """Two local jobs on one machine must not collide on 29500."""
        seen_ports = []

        def fake_popen(cmd, env=None, **kw):
            seen_ports.append(env["MASTER_PORT"])
            p = MagicMock()
            p.poll.return_value = 0
            p.pid = 1
            return p

        monkeypatch.setattr("autotrainer.launcher.subprocess.Popen", fake_popen)
        monkeypatch.setattr("autotrainer.launcher.time.sleep", lambda *_: None)
        monkeypatch.setattr(
            "autotrainer.launcher.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=2, gpus=2),
        )
        assert launch("dummy_script.py", []) == 0
        assert len(set(seen_ports)) == 1  # all workers share one rendezvous
        assert seen_ports[0] != "29500"  # OS-assigned, not the collision-prone default

    def test_explicit_autotrainer_port_is_pinned(self, monkeypatch):
        seen_ports = []

        def fake_popen(cmd, env=None, **kw):
            seen_ports.append(env["MASTER_PORT"])
            p = MagicMock()
            p.poll.return_value = 0
            p.pid = 1
            return p

        monkeypatch.setenv("AUTOTRAINER_PORT", "40123")
        monkeypatch.setattr("autotrainer.launcher.subprocess.Popen", fake_popen)
        monkeypatch.setattr("autotrainer.launcher.time.sleep", lambda *_: None)
        monkeypatch.setattr(
            "autotrainer.launcher.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=1, gpus=2),
        )
        assert launch("dummy_script.py", []) == 0
        assert seen_ports == ["40123"]

    def test_keyboard_interrupt_returns_130(self, monkeypatch):
        p = MagicMock()
        p.poll.side_effect = [None, KeyboardInterrupt()]
        p.pid = 1
        monkeypatch.setattr("autotrainer.launcher.subprocess.Popen", lambda *a, **kw: p)
        monkeypatch.setattr("autotrainer.launcher.time.sleep", lambda *_: None)
        monkeypatch.setattr(
            "autotrainer.launcher.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=1, gpus=2),
        )
        assert launch("dummy_script.py", []) == 130

    def test_each_child_pinned_to_its_own_gpu(self, monkeypatch):
        """Per-child CUDA_VISIBLE_DEVICES isolates GPUs (torchrun pattern):
        each worker sees exactly one GPU so device 0 == its assigned GPU,
        instead of all workers seeing all GPUs and racing on set_device."""
        seen_cvd = []

        def fake_popen(cmd, env=None, **kw):
            seen_cvd.append(env["CUDA_VISIBLE_DEVICES"])
            p = MagicMock()
            p.poll.return_value = 0
            p.pid = 1
            return p

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr("autotrainer.launcher.subprocess.Popen", fake_popen)
        monkeypatch.setattr("autotrainer.launcher.time.sleep", lambda *_: None)
        monkeypatch.setattr(
            "autotrainer.launcher.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=4, gpus=4),
        )
        assert launch("dummy_script.py", []) == 0
        # No CUDA_VISIBLE_DEVICES set by the user -> each local_rank maps to
        # itself: worker 0 sees GPU 0, worker 1 sees GPU 1, etc.
        assert seen_cvd == ["0", "1", "2", "3"]

    def test_child_gpu_isolation_honors_user_cvd(self, monkeypatch):
        """If the user restricted CUDA_VISIBLE_DEVICES="2,3", the spawned
        workers must be pinned to those physical GPUs (in order), not to the
        raw local_rank indices."""
        seen_cvd = []

        def fake_popen(cmd, env=None, **kw):
            seen_cvd.append(env["CUDA_VISIBLE_DEVICES"])
            p = MagicMock()
            p.poll.return_value = 0
            p.pid = 1
            return p

        monkeypatch.setattr("autotrainer.launcher.subprocess.Popen", fake_popen)
        monkeypatch.setattr("autotrainer.launcher.time.sleep", lambda *_: None)
        monkeypatch.setattr(
            "autotrainer.launcher.detect",
            lambda: Environment(mode="local_multi_gpu", nproc_per_node=2, gpus=2),
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert launch("dummy_script.py", []) == 0
        assert seen_cvd == ["2", "3"]
