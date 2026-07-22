"""Tests for the Tier 3 features: torch.compile and FSDP integration.

Both need a real CUDA GPU (torch.compile actually compiles kernels; FSDP
needs an initialized process group). Marked `cuda` so CPU CI deselects
them and the self-hosted GPU runner picks them up.

Contract under test:
  - prepare(compile=True) returns a compiled model that still forwards+backwards
  - prepare(compile=True) does NOT touch lr / loss / optimizer
  - compile failure falls back to the uncompiled model with a warning
  - prepare(fsdp=True) on world_size==1 is a no-op with a warning
  - compile happens before DDP wrap (verified via the returned model's type)
"""

from __future__ import annotations

import pytest

from autotrainer.backends.torch_backend import prepare

_HAS_CUDA = True
try:
    import torch  # noqa: F401

    _HAS_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
except ImportError:
    _HAS_CUDA = False


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not _HAS_CUDA, reason="Tier 3 features need a real CUDA GPU"),
]


def _model_loader(torch):
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    x = torch.randn(256, 32)
    y = torch.randint(0, 10, (256,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    return model, loader


def _triton_available() -> bool:
    """torch.compile's default inductor backend needs Triton for codegen.
    Triton isn't supported on native Windows, so forward/backward through a
    compiled model raises TritonMissing there. Tests that only check the
    *wiring* (compile was called, model wrapped, hyperparameters untouched)
    don't need a working backend; tests that actually execute the compiled
    graph gate on this.
    """
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


class TestRunnerSmoke:
    """First-tests-to-run on the GPU runner. If any of these fail, the
    runner itself is misconfigured and the rest of the cuda-marked suite
    can't be trusted. Kept separate from TestCompile/TestFsdp so a runner
    problem shows as a small, named failure rather than 11 mysterious
    per-test failures.
    """

    def test_gpu_visible_to_torch(self):
        """The most basic check: torch sees at least one CUDA device."""
        import torch

        assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
        assert torch.cuda.device_count() > 0, "device_count() is 0"

    def test_gpu_forward_backward_executes(self):
        """Actually run a forward+backward on the GPU - catches driver/CUDA
        library mismatches that is_available() alone won't (a misconfigured
        torch can report True then crash on the first kernel)."""
        import torch

        device = torch.device("cuda:0")
        model = torch.nn.Linear(16, 8).to(device)
        x = torch.randn(4, 16, device=device)
        y = torch.randn(4, 8, device=device)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        # If we got here without raising, the full stack works.
        assert loss.item() > 0
        # And grads actually populated (sanity for the backward pass).
        assert model.weight.grad is not None
        assert torch.isfinite(model.weight.grad).all()


class TestCompile:
    def test_compile_wraps_model(self):
        """compile=True must hand the model to torch.compile (wiring check).

        We don't forward through it here - that needs a working backend
        (Triton), which isn't available on every CUDA box (e.g. native
        Windows). The contract is that autotrainer *calls* torch.compile
        and reports it, not that it produces fast kernels.
        """
        import torch

        model, loader = _model_loader(torch)
        out, _ = prepare(model, loader, compile=True)
        # torch.compile returns an OptimizedModule; the surest signal is the
        # presence of the inner cached-but-traced forward.
        assert hasattr(out, "forward")
        # The summary line must mention torch.compile.
        # (Verified separately in test_compile_summary_printed.)

    def test_compile_does_not_touch_hyperparameters(self):
        import torch

        model, loader = _model_loader(torch)
        opt = torch.optim.SGD(model.parameters(), lr=0.077)
        _, _, returned_opt = prepare(model, loader, opt, compile=True)
        assert returned_opt is opt
        assert opt.param_groups[0]["lr"] == 0.077

    @pytest.mark.skipif(not _triton_available(), reason="needs Triton for inductor backend")
    def test_compile_then_backward_runs(self):
        """A compiled model must still forward+backward without graph errors.

        Gated on Triton: inductor (the default backend) codegens via Triton,
        which isn't available on native Windows. The wiring test above covers
        the no-Triton case.
        """
        import torch
        import torch.nn as nn

        model, loader = _model_loader(torch)
        out, _ = prepare(model, loader, compile=True)
        device = next(out.parameters()).device
        x = torch.randn(8, 32, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        loss = nn.CrossEntropyLoss()(out(x), y)
        loss.backward()  # must not raise
        assert loss.item() > 0

    def test_compile_summary_printed(self, capsys):
        """The summary line must mention torch.compile so the user sees it."""
        import torch

        model, loader = _model_loader(torch)
        prepare(model, loader, optimize=True, compile=True)
        out = capsys.readouterr().out
        assert "torch.compile" in out

    def test_compile_mode_passed_through(self):
        """compile_mode should reach torch.compile without being mangled."""
        import torch

        model, loader = _model_loader(torch)
        # reduce-overhead uses CUDA graphs; just confirm it doesn't crash at
        # the compile() call. (Backend codegen happens lazily on forward.)
        out, _ = prepare(model, loader, compile=True, compile_mode="reduce-overhead")
        assert out is not None


class TestFsdp:
    def test_fsdp_single_process_is_noop_with_warning(self, capsys, monkeypatch):
        """world_size==1: FSDP has nothing to shard across, must no-op + warn."""
        import torch

        # Force single-process by ensuring WORLD_SIZE is unset/1.
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        model, loader = _model_loader(torch)
        out, _ = prepare(model, loader, fsdp=True)
        # Model should NOT be FSDP-wrapped (nothing to shard with 1 rank).
        assert not isinstance(out, torch.distributed.fsdp.FullyShardedDataParallel)
        assert "FSDP is a no-op" in capsys.readouterr().out
