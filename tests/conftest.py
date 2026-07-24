"""Shared pytest fixtures.

`clean_env` strips the distributed/SLURM environment variables between tests so
one test's env mutation can't leak into the next. It is autouse so every test
in every file gets it without importing anything.

`pretend_cuda` patches the CUDA entry points the optimize path reads, on the
*real* ``torch.cuda`` module object (not a wholesale fake - torch's own import
machinery reads ``torch.cuda`` internals, and replacing the whole module
corrupts its lazy init). The set of patched attributes is the single source
of truth for the CUDA surface ``prepare()`` touches; a guard test
(``test_optimize.py::TestCudaSurfaceCoverage``) fails loudly if ``prepare()``
starts reading a public ``torch.cuda`` attribute that isn't on this list,
closing the "silently exercises the wrong path" gap that bit PR #1.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip SLURM/rank vars so tests don't leak into each other."""
    for k in list(os.environ):
        if k.startswith(("SLURM_", "RANK", "LOCAL_RANK", "WORLD_SIZE", "AUTOTRAINER_")):
            monkeypatch.delenv(k, raising=False)


# The complete public ``torch.cuda`` surface the optimize path reads. This is
# the single source of truth: ``pretend_cuda`` patches every name here, and
# test_optimize.py::TestCudaSurfaceCoverage asserts the source under test
# doesn't reach for a public torch.cuda.* name that isn't listed. When you add
# a new CUDA call to prepare()/_optimize/utils, add the name here too (and a
# fake in pretend_cuda) so the stub stays honest.
CUDA_OPTIMIZE_SURFACE = frozenset(
    {
        "is_available",
        "device_count",
        "set_device",
        "current_device",
        "is_bf16_supported",
        "get_device_name",
        "get_device_properties",
        "max_memory_reserved",
        "empty_cache",
        "synchronize",
    }
)


@pytest.fixture
def pretend_cuda(monkeypatch):
    """Make the real ``torch.cuda`` report a single fake GPU for one test.

    Patches the known public surface (CUDA_OPTIMIZE_SURFACE) *on the real
    module object* so torch's internal ``torch.cuda`` reads still work, and
    makes ``Tensor.to(device)`` a no-op so model placement doesn't hit the
    unbuilt CUDA extension. Restored automatically by monkeypatch.

    Returns a SimpleNamespace holding the fake values a test may want to tune
    (e.g. ``device_name`` for the throughput MFU table).
    """
    import torch

    fakes = SimpleNamespace(device_name="A100")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _d: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _d=0: fakes.device_name)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _d=0: SimpleNamespace(
            name=fakes.device_name, major=8, minor=0, total_memory=80 * 1024**3
        ),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _d=None: 10 * 1024**3)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _d=None: None)
    # Device placement is orthogonal to what the optimize tests check (flag
    # application + non-mutation of hyperparameters); keep tensors where they
    # are so "to(cuda)" becomes a no-op.
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)
    return fakes
