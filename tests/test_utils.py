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
