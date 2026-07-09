"""Test suite. Run with: pytest tests/ -v"""

import os
import subprocess
import sys

import pytest

import autotrainer
from autotrainer.detect import detect


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip SLURM/rank vars so tests don't leak into each other."""
    for k in list(os.environ):
        if k.startswith(("SLURM_", "RANK", "LOCAL_RANK", "WORLD_SIZE", "AUTOTRAINER_")):
            monkeypatch.delenv(k, raising=False)


class TestDetect:
    def test_single_mode_default(self):
        env = detect()
        assert env.mode in ("single", "local_multi_gpu")
        assert env.world_size >= 1

    def test_slurm_detection(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "1")
        monkeypatch.setenv("SLURM_NNODES", "3")
        monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
        monkeypatch.setenv("SLURM_NODEID", "1")
        monkeypatch.setenv("SLURM_NODELIST", "n01,n02,n03")
        env = detect()
        assert env.mode == "slurm"
        assert env.nnodes == 3
        assert env.world_size == 6
        assert env.node_rank == 1

    def test_slurm_cpu_only_falls_back_to_ntasks(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "1")
        monkeypatch.setenv("SLURM_NNODES", "2")
        monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "4")
        env = detect()
        assert env.nproc_per_node == 4


class TestDispatcher:
    def test_sklearn_routing(self):
        sklearn = pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        rf = autotrainer.prepare(RandomForestClassifier())
        assert rf.n_jobs >= 1

    def test_xgboost_routes_before_sklearn(self):
        xgboost = pytest.importorskip("xgboost")
        from xgboost import XGBClassifier
        m = autotrainer.prepare(XGBClassifier())
        assert m.n_jobs >= 1

    def test_slurm_cpu_allocation_respected(self, monkeypatch):
        pytest.importorskip("sklearn")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
        from sklearn.ensemble import RandomForestClassifier
        rf = autotrainer.prepare(RandomForestClassifier())
        assert rf.n_jobs == 6

    def test_unknown_model_raises(self):
        with pytest.raises(TypeError):
            autotrainer.prepare(object())


class TestBoostParams:
    def test_xgb_key(self):
        p = autotrainer.boost_params({"eta": 0.1})
        assert "nthread" in p and p["eta"] == 0.1

    def test_lgbm_key(self):
        p = autotrainer.boost_params(lib="lightgbm")
        assert "num_threads" in p


class TestTFConfig:
    def test_tf_config_shape(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODELIST", "a01,a02")
        monkeypatch.setenv("SLURM_NODEID", "1")
        from autotrainer.backends.tf_backend import build_tf_config
        cfg = build_tf_config(port=12345)
        assert cfg["cluster"]["worker"] == ["a01:12345", "a02:12345"]
        assert cfg["task"] == {"type": "worker", "index": 1}


class TestUtils:
    def test_rank_zero_helpers(self, monkeypatch, capsys):
        monkeypatch.setenv("RANK", "0")
        assert autotrainer.is_main()
        autotrainer.print0("hello")
        assert "hello" in capsys.readouterr().out

    def test_nonzero_rank_is_silent(self, monkeypatch, capsys):
        monkeypatch.setenv("RANK", "2")
        assert not autotrainer.is_main()
        autotrainer.print0("hello")
        assert capsys.readouterr().out == ""

    def test_barrier_noop_without_dist(self):
        autotrainer.barrier()  # must not raise


class TestCLI:
    def test_info_runs(self):
        r = subprocess.run([sys.executable, "-m", "autotrainer.cli", "info"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "mode" in r.stdout

    def test_doctor_runs(self):
        r = subprocess.run([sys.executable, "-m", "autotrainer.cli", "doctor"],
                           capture_output=True, text=True)
        assert "detected mode" in r.stdout
