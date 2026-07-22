"""Tests for Tier 3 #14 (cpu_offload) and #15 (SLURM node-local scratch).

cpu_offload only takes effect on the multi-rank FSDP path, which needs a
real process group - so we test the single-process and DDP-fallback
behaviors (the no-op-with-warning paths) and the env-var plumbing. The
multi-rank FSDP+offload path needs the same harness as the existing
distributed tests; out of scope here.

SLURM scratch is mostly env-var logic and runs anywhere (no GPU needed),
so those tests are NOT cuda-marked.
"""

from __future__ import annotations

import os

import pytest

from autotrainer import configure_scratch, node_scratch
from autotrainer.backends.torch_backend import prepare

# --------------------------------------------------------------------------- #
# CPU offload (Tier 3 #14)
# --------------------------------------------------------------------------- #


_HAS_CUDA = True
try:
    import torch  # noqa: F401

    _HAS_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
except ImportError:
    _HAS_CUDA = False


def _model_loader(torch):
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    x = torch.randn(256, 32)
    y = torch.randint(0, 10, (256,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    return model, loader


@pytest.mark.cuda
@pytest.mark.skipif(not _HAS_CUDA, reason="cpu_offload wiring needs CUDA prepare()")
class TestCpuOffload:
    def test_cpu_offload_ignored_without_fsdp(self, capsys, monkeypatch):
        """cpu_offload=True without fsdp=True on the DDP path must warn + ignore."""
        import torch

        # Force single-process so we don't need a process group.
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        model, loader = _model_loader(torch)
        # Single-process path: fsdp is a no-op, cpu_offload must warn too.
        prepare(model, loader, fsdp=False, cpu_offload=True)
        out = capsys.readouterr().out
        assert "cpu_offload" in out
        assert "ignored" in out

    def test_cpu_offload_ignored_with_fsdp_single_process(self, capsys, monkeypatch):
        """On world_size==1, FSDP is a no-op, so cpu_offload must be too."""
        import torch

        monkeypatch.delenv("WORLD_SIZE", raising=False)
        model, loader = _model_loader(torch)
        prepare(model, loader, fsdp=True, cpu_offload=True)
        out = capsys.readouterr().out
        assert "FSDP is a no-op" in out
        assert "cpu_offload" in out


# --------------------------------------------------------------------------- #
# SLURM node-local scratch (Tier 3 #15)
# --------------------------------------------------------------------------- #


class TestNodeScratch:
    def test_outside_slurm_uses_system_temp(self, monkeypatch, tmp_path):
        # No SLURM_JOB_ID -> system temp dir, not an NFS-looking path.
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        scratch = node_scratch()
        assert scratch.exists()
        assert "autotrainer" in str(scratch)

    def test_under_slurm_includes_job_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        scratch = node_scratch()
        assert scratch.exists()
        assert "12345" in str(scratch)

    def test_creates_directory_if_missing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        nested = tmp_path / "deep" / "nested"
        monkeypatch.setenv("TMPDIR", str(nested))
        # node_scratch() must mkdir -p the base before returning.
        scratch = node_scratch()
        assert scratch.exists()


class TestConfigureScratch:
    def test_sets_torchinductor_cache_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        configure_scratch(warn=False)
        cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
        assert "inductor" in cache
        assert "autotrainer" in cache

    def test_does_not_overwrite_existing_cache_dir(self, monkeypatch, tmp_path):
        # setdefault semantics: a user-set value wins.
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/my/custom/cache")
        configure_scratch(warn=False)
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/my/custom/cache"

    def test_warns_when_scratch_looks_networked(self, monkeypatch, capsys, tmp_path):
        # Force the heuristic to True (real path-based detection is covered
        # by test_looks_networked_pure_logic below). Using a real tmp_path
        # avoids PermissionError when node_scratch() tries to mkdir at an
        # unusual path on locked-down CI runners (e.g. /nfs on Ubuntu, where
        # creating /nfs/... from root requires write perms the runner lacks).
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        import autotrainer.slurm as slurm_mod

        monkeypatch.setattr(slurm_mod, "_looks_networked", lambda _p: True)
        configure_scratch(warn=True)
        out = capsys.readouterr().out
        assert "network filesystem" in out

    def test_no_warning_when_scratch_is_local(self, monkeypatch, capsys, tmp_path):
        # Symmetric: when the scratch is local, no network warning fires.
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        configure_scratch(warn=True)
        out = capsys.readouterr().out
        assert "network filesystem" not in out

    def test_looks_networked_pure_logic(self):
        """The heuristic itself, OS-independent."""
        from pathlib import Path

        from autotrainer.slurm import _looks_networked

        assert _looks_networked(Path("/home/user"))
        assert _looks_networked(Path("/nfs/scratch"))
        assert _looks_networked(Path("/scratch/lustre/job42"))
        assert _looks_networked(Path("/panfs/data"))
        assert not _looks_networked(Path("/tmp"))
        assert not _looks_networked(Path("/var/tmp"))

    def test_returns_scratch_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        scratch = configure_scratch(warn=False)
        assert scratch.exists()
