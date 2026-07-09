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


class TestPytorchEnhancements:
    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_to_device(self):
        import torch
        from autotrainer.utils import to_device
        device = torch.device("cpu")
        x = torch.tensor([1, 2, 3])
        data = {
            "tensor": x,
            "list": [x, 42],
            "tuple": (x, "hello"),
            "nested": {"tensor": x}
        }
        res = to_device(data, device)
        assert torch.equal(res["tensor"], x)
        assert torch.equal(res["list"][0], x)
        assert res["list"][1] == 42
        assert torch.equal(res["tuple"][0], x)
        assert res["tuple"][1] == "hello"
        assert torch.equal(res["nested"]["tensor"], x)

    def test_slice_batch(self):
        import torch
        from autotrainer.utils import slice_batch
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        data = {
            "tensor": x,
            "list": [x, 42],
            "tuple": (x, "hello")
        }
        res = slice_batch(data, 2)
        assert res["tensor"].shape[0] == 2
        assert res["list"][0].shape[0] == 2
        assert res["tuple"][0].shape[0] == 2

    def test_robust_forward(self):
        import torch
        import torch.nn as nn
        from autotrainer.utils import robust_forward

        class DictModel(nn.Module):
            def forward(self, x, y=None):
                return x + (y if y is not None else 0)

        class ListModel(nn.Module):
            def forward(self, *args):
                return sum(args)

        dm = DictModel()
        lm = ListModel()
        
        assert robust_forward(dm, {"x": 5, "y": 10}) == 15
        assert robust_forward(dm, {"x": 5}) == 5
        assert robust_forward(lm, [1, 2, 3]) == 6
        assert robust_forward(lm, (4, 5)) == 9
        assert robust_forward(dm, 7) == 7

    def test_grad_scaler(self):
        import torch
        from autotrainer.utils import GradScaler
        scaler = GradScaler()
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            assert isinstance(scaler, (torch.amp.GradScaler, torch.cuda.amp.GradScaler))
        else:
            assert isinstance(scaler, torch.cuda.amp.GradScaler)
        expected_enabled = False
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            expected_enabled = True
        assert scaler.is_enabled() == expected_enabled

    def test_find_lr_with_dict(self):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from autotrainer.auto_optim import find_lr

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 1)
            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.x = torch.randn(50, 5)
                self.y = torch.randn(50, 1)
            def __len__(self):
                return len(self.x)
            def __getitem__(self, idx):
                return {"x": self.x[idx]}, self.y[idx]

        loader = DataLoader(DictDataset(), batch_size=5)
        loss_fn = nn.MSELoss()
        
        best_lr = find_lr(model, loader, loss_fn, min_lr=1e-5, max_lr=1e-1, num_iters=10)
        assert isinstance(best_lr, float)
        assert best_lr > 0

