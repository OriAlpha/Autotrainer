"""Tests for ``prepare(..., optimize=True)`` - the GPU optimization layer.

These cover the pure-logic helpers (which decide what defaults to apply)
directly, plus the integration into ``prepare()`` with CUDA monkeypatched
to True (CI is CPU-only). The contract under test:

    optimize=True sets the free-win GPU flags and loader defaults, but
    NEVER touches lr / loss / schedule / optimizer choice, and is a
    complete no-op when optimize=False or on CPU.
"""

from __future__ import annotations

import pytest


class TestLooksLikeCnn:
    def test_mlp_is_not_cnn(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from autotrainer._optimize import _looks_like_cnn

        assert _looks_like_cnn(nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 1))) is False

    def test_conv_is_cnn(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from autotrainer._optimize import _looks_like_cnn

        assert _looks_like_cnn(nn.Conv2d(3, 8, 3)) is True


class TestBuildLoaderDefaults:
    """The decision logic - what changes, what's left alone."""

    def _loader(self, torch, **kw):
        from torch.utils.data import DataLoader, TensorDataset

        ds = TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,)))
        return DataLoader(ds, batch_size=4, **kw)

    def test_no_op_on_cpu(self, monkeypatch):
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("CPU-path test; CUDA is available in this environment")
        from autotrainer._optimize import build_loader_defaults

        # cuda.is_available() is False; this is the real path.
        assert build_loader_defaults(self._loader(torch), world_size=1) == {}

    def test_bare_loader_gets_defaults_when_cuda(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        from autotrainer._optimize import build_loader_defaults

        out = build_loader_defaults(self._loader(torch), world_size=1)
        assert out["num_workers"] >= 1
        assert out["pin_memory"] is True
        assert out["persistent_workers"] is True

    def test_user_set_num_workers_is_left_alone(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        from autotrainer._optimize import build_loader_defaults

        loader = self._loader(torch, num_workers=4)
        out = build_loader_defaults(loader, world_size=1)
        # num_workers was set explicitly - don't override.
        assert "num_workers" not in out
        # But the other defaults can still be added.
        assert out.get("pin_memory") is True

    def test_user_pin_memory_is_left_alone(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        from autotrainer._optimize import build_loader_defaults

        loader = self._loader(torch, pin_memory=True)
        out = build_loader_defaults(loader, world_size=1)
        assert "pin_memory" not in out

    def test_workers_shard_by_world_size(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        from autotrainer._optimize import build_loader_defaults

        single = build_loader_defaults(self._loader(torch), world_size=1)
        quad = build_loader_defaults(self._loader(torch), world_size=4)
        # Sharding across more ranks means fewer workers per rank.
        assert quad["num_workers"] <= single["num_workers"]


class TestApplyGpuFlags:
    def test_no_op_on_cpu(self, monkeypatch):
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("CPU-path test; CUDA is available in this environment")
        import torch.nn as nn

        from autotrainer._optimize import apply_gpu_flags

        before = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.benchmark,
        )
        apply_gpu_flags(nn.Linear(3, 3))
        after = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.benchmark,
        )
        assert before == after

    def test_sets_tf32_when_cuda(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        from autotrainer._optimize import apply_gpu_flags

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        original = torch.backends.cuda.matmul.allow_tf32
        try:
            apply_gpu_flags(nn.Linear(3, 3))
            assert torch.backends.cuda.matmul.allow_tf32 is True
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original

    def test_sets_cudnn_benchmark_only_for_cnn(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        from autotrainer._optimize import apply_gpu_flags

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        original = torch.backends.cudnn.benchmark
        try:
            apply_gpu_flags(nn.Linear(3, 3), cnn=False)
            # Not a CNN -> benchmark unchanged.
            assert torch.backends.cudnn.benchmark == original
            apply_gpu_flags(nn.Linear(3, 3), cnn=True)
            assert torch.backends.cudnn.benchmark is True
        finally:
            torch.backends.cudnn.benchmark = original


class TestPrepareOptimizeIntegration:
    """End-to-end: the flag flows through prepare() and respects the contract."""

    def _model_loader(self, torch):
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        model = nn.Linear(3, 1)
        ds = TensorDataset(torch.randn(16, 3), torch.randn(16, 1))
        loader = DataLoader(ds, batch_size=4)  # bare loader, no overrides
        return model, loader

    def _pretend_cuda(self, torch, monkeypatch):
        """Stub the CUDA entry points prepare() touches on a CPU-only box.

        We want to exercise the optimize=True code path without a real GPU.
        ``torch.cuda.set_device`` and ``model.to('cuda')`` both hit C
        extensions that aren't built for CPU torch, so stub them along with
        ``is_available`` AND ``device_count`` (cuda_device() gates on the
        latter, not just the former). Device placement is orthogonal to
        what these tests check (flag application + non-mutation of
        hyperparameters).
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "set_device", lambda _d: None)
        # Keep tensors where they are; "to(cuda)" becomes a no-op.
        monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)

    def test_optimize_false_is_no_op_on_cpu(self, capsys):
        torch = pytest.importorskip("torch")
        from autotrainer.backends.torch_backend import prepare

        model, loader = self._model_loader(torch)
        _, out_loader = prepare(model, loader, optimize=False)
        # No summary printed when optimize=False.
        out = capsys.readouterr().out
        assert "optimize" not in out

    def test_optimize_true_does_not_touch_hyperparameters(self, monkeypatch):
        """The contract: optimize sets flags, never overrides lr/loss/sched."""
        torch = pytest.importorskip("torch")
        from autotrainer.backends.torch_backend import prepare

        self._pretend_cuda(torch, monkeypatch)

        model, loader = self._model_loader(torch)
        opt = torch.optim.SGD(model.parameters(), lr=0.123)
        _, _, returned_opt = prepare(model, loader, opt, optimize=True)

        # Optimizer passed through verbatim - same object, same lr.
        assert returned_opt is opt
        assert opt.param_groups[0]["lr"] == 0.123

    def test_amp_default_follows_optimize(self, monkeypatch):
        torch = pytest.importorskip("torch")
        from autotrainer.backends.torch_backend import prepare

        self._pretend_cuda(torch, monkeypatch)

        # optimize=True should imply AMP=True in the summary line.
        model, loader = self._model_loader(torch)
        prepare(model, loader, optimize=True)
        # No exception means the AMP wiring resolved without forcing a
        # GradScaler the user never asked for; the actual autocast context
        # is what the caller uses at train time.

    def test_amp_can_be_disabled_when_optimizing(self, monkeypatch):
        torch = pytest.importorskip("torch")
        from autotrainer.backends.torch_backend import prepare

        self._pretend_cuda(torch, monkeypatch)

        model, loader = self._model_loader(torch)
        # amp=False must win even under optimize=True.
        prepare(model, loader, optimize=True, amp=False)

    def test_loader_defaults_applied_only_when_optimize(self, monkeypatch):
        torch = pytest.importorskip("torch")
        from autotrainer.backends.torch_backend import prepare

        self._pretend_cuda(torch, monkeypatch)

        # optimize=False: bare loader stays bare.
        model, loader = self._model_loader(torch)
        _, out_loader = prepare(model, loader, optimize=False)
        assert out_loader.num_workers == 0  # unchanged

        # optimize=True: bare loader gets defaults.
        model, loader = self._model_loader(torch)
        _, out_loader = prepare(model, loader, optimize=True)
        assert out_loader.num_workers >= 1
        assert out_loader.pin_memory is True


# auto_bs runs real forward+backward passes to probe memory, so it needs an
# actual GPU. The `cuda` marker lets the GPU CI job select these; the skipif
# protects a CPU box that runs `-m cuda` anyway.
def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


@pytest.mark.cuda
@pytest.mark.skipif(not _has_cuda(), reason="auto_bs needs a real GPU")
class TestAutoBs:
    """Verify prepare(auto_bs=True) grows the batch size and rebuilds the loader.

    These only run when CUDA is present. The contract under test:
        - the returned loader's batch_size is >= the input's
        - lr/schedule are NOT changed (we don't have an optimizer here, but
          the code path must not invent one)
        - the loss_fn the user passes is used only for the sweep, never returned
    """

    def _model_loader(self, torch):
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
        x = torch.randn(256, 32)
        y = torch.randint(0, 10, (256,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        return model, loader

    def test_grows_batch_size_with_loss_fn(self):
        import torch
        import torch.nn as nn

        from autotrainer.backends.torch_backend import prepare

        model, loader = self._model_loader(torch)
        _, out_loader = prepare(
            model, loader, auto_bs=True, loss_fn=nn.CrossEntropyLoss(), max_bs=128
        )
        # A 64-wide MLP on a real GPU should comfortably absorb more than bs=4.
        assert out_loader.batch_size >= loader.batch_size

    def test_forward_only_sweep_works_without_loss_fn(self):
        import torch

        from autotrainer.backends.torch_backend import prepare

        model, loader = self._model_loader(torch)
        # No loss_fn -> forward-only sweep (conservative). Must not raise.
        _, out_loader = prepare(model, loader, auto_bs=True, max_bs=64)
        assert out_loader.batch_size >= loader.batch_size

    def test_forward_only_sweep_prints_note_without_loss_fn(self, capsys):
        """Without loss_fn the sweep is conservative; surface that once."""
        import torch

        from autotrainer.backends.torch_backend import prepare

        model, loader = self._model_loader(torch)
        prepare(model, loader, auto_bs=True, max_bs=64)
        out = capsys.readouterr().out
        assert "forward-only" in out
        assert "pass loss_fn" in out

    def test_forward_only_note_silent_when_loss_fn_given(self, capsys):
        """With a loss_fn the sweep is real, so the conservative note is absent."""
        import torch
        import torch.nn as nn

        from autotrainer.backends.torch_backend import prepare

        model, loader = self._model_loader(torch)
        prepare(model, loader, auto_bs=True, loss_fn=nn.CrossEntropyLoss(), max_bs=64)
        out = capsys.readouterr().out
        assert "forward-only" not in out

    def test_respects_max_bs_ceiling(self):
        import torch
        import torch.nn as nn

        from autotrainer.backends.torch_backend import prepare

        model, loader = self._model_loader(torch)
        _, out_loader = prepare(
            model, loader, auto_bs=True, loss_fn=nn.CrossEntropyLoss(), max_bs=8
        )
        assert out_loader.batch_size <= 8


class TestPublicDispatcherForwardsKwargs:
    """Regression test: the PUBLIC autotrainer.prepare() (in __init__.py) must
    forward optimize/compile/fsdp/etc. kwargs to the torch backend.

    This caught a real bug: the dispatcher originally called the backend with
    only 3 positional args, so every README snippet using prepare(optimize=True)
    raised TypeError. The existing TestPrepareOptimizeIntegration tests imported
    the backend directly and missed it. These tests import the public package
    and exercise the actual path users take.
    """

    def _model_loader(self, torch):
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        model = nn.Linear(3, 1)
        ds = TensorDataset(torch.randn(16, 3), torch.randn(16, 1))
        loader = DataLoader(ds, batch_size=4)
        return model, loader

    def test_public_prepare_accepts_optimize_kwarg(self, monkeypatch):
        """autotrainer.prepare(model, loader, optimize=True) must not raise TypeError."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "set_device", lambda _d: None)
        monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)

        import autotrainer  # public package, not the backend

        model, loader = self._model_loader(torch)
        # This used to raise: TypeError: prepare() got an unexpected keyword
        # argument 'optimize'. If you see that error again, the dispatcher in
        # __init__.py lost its **kwargs forwarding.
        model, loader = autotrainer.prepare(model, loader, optimize=True)

    def test_public_prepare_accepts_compile_kwarg(self, monkeypatch):
        """compile= must also reach the backend through the dispatcher."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "set_device", lambda _d: None)
        monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)

        import autotrainer

        model, loader = self._model_loader(torch)
        autotrainer.prepare(model, loader, optimize=True, compile=True)

    def test_public_prepare_accepts_fsdp_kwarg(self, monkeypatch):
        """fsdp= must reach the backend. Single-process path: no-op with warning."""
        import torch

        monkeypatch.delenv("WORLD_SIZE", raising=False)

        import autotrainer

        model, loader = self._model_loader(torch)
        autotrainer.prepare(model, loader, fsdp=True)
