"""Tests for utils.py: rank-aware helpers and mixed-precision utilities."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import contextlib  # noqa: E402

from autotrainer.utils import (  # noqa: E402
    autocast_context,
    get_batch_size,
    get_model_device,
    save0,
    set_epoch,
)


class TestSave0:
    def test_save0_writes_on_rank0(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        path = tmp_path / "tensors.pt"
        save0({"w": torch.tensor([1.0, 2.0])}, str(path))
        assert path.exists()
        loaded = torch.load(path, weights_only=True)
        assert torch.equal(loaded["w"], torch.tensor([1.0, 2.0]))

    def test_save0_skips_on_nonzero_rank(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        path = tmp_path / "should_not_exist.pt"
        save0({"w": torch.tensor([1.0])}, str(path))
        assert not path.exists()


class TestAutocastContext:
    def test_returns_context_manager(self):
        ctx = autocast_context()
        # Must be usable as a `with` block regardless of CUDA availability.
        with ctx:
            pass

    def test_is_nullcontext_on_cpu(self):
        if not torch.cuda.is_available():
            assert isinstance(autocast_context(), contextlib.nullcontext)


class TestGetBatchSize:
    def test_finds_leading_dim_in_dict(self):
        data = {"x": torch.randn(8, 3), "y": torch.randn(8, 1)}
        assert get_batch_size(data) == 8

    def test_finds_in_nested_list(self):
        data = [torch.randn(4, 2), torch.randn(4)]
        assert get_batch_size(data) == 4

    def test_returns_zero_for_scalars(self):
        assert get_batch_size(42) == 0


class TestSetEpoch:
    def test_forwards_to_distributed_sampler(self):
        from torch.utils.data import DataLoader, TensorDataset
        from torch.utils.data.distributed import DistributedSampler

        ds = TensorDataset(torch.randn(8, 3))
        sampler = DistributedSampler(ds, num_replicas=2, rank=0)
        loader = DataLoader(ds, batch_size=2, sampler=sampler)
        set_epoch(loader, 5)
        assert sampler.epoch == 5

    def test_reshuffles_between_epochs(self):
        # The whole point: without set_epoch the order repeats every epoch.
        from torch.utils.data import TensorDataset
        from torch.utils.data.distributed import DistributedSampler

        ds = TensorDataset(torch.arange(64))
        sampler = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=True)

        class _Loader:
            pass

        loader = _Loader()
        loader.sampler = sampler
        set_epoch(loader, 0)
        order0 = list(sampler)
        set_epoch(loader, 1)
        assert list(sampler) != order0

    def test_noop_for_plain_loader(self):
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(TensorDataset(torch.randn(4, 3)), batch_size=2)
        set_epoch(loader, 3)  # SequentialSampler has no set_epoch; must not raise

    def test_noop_for_object_without_sampler(self):
        set_epoch(object(), 1)  # must not raise


class TestGetModelDevice:
    def test_defaults_to_cpu_for_paramless_model(self):
        # A Module with no parameters falls back to CPU.
        import torch.nn as nn

        class Empty(nn.Module):
            pass

        assert get_model_device(Empty()) == torch.device("cpu")

    def test_returns_param_device(self):
        import torch.nn as nn

        model = nn.Linear(3, 2)
        assert get_model_device(model) == torch.device("cpu")
