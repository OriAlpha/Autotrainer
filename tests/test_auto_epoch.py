"""The auto-epoch DataLoader wrapper must survive everything prepare() does.

``_shard_loader`` wraps the sharded loader in ``_AutoEpochDataLoader``, which
calls ``DistributedSampler.set_epoch(epoch)`` on each ``iter()`` so every epoch
draws a fresh permutation. That wrapper shipped with no test at all, and
``prepare()`` then dropped it: both the ``optimize`` loader-defaults step and
the ``auto_bs`` sweep construct a *fresh* ``DataLoader`` from the current one's
dataset/sampler/kwargs, which a wrapper instance does not survive.

The result was silent and specific to the most common setup there is -
multi-GPU, ``optimize=True`` (the default), and a loader the user never tuned,
since ``build_loader_defaults()`` only returns work to do when ``num_workers``
is 0 or ``pin_memory`` is off. ``set_epoch`` was never called, so every epoch
replayed one fixed permutation on every rank. Nothing raised; the loss curve
just quietly got worse than it should have been.

These tests pin the wrapper's survival at the boundary that broke it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from autotrainer.backends import torch_backend  # noqa: E402
from autotrainer.backends.torch_backend import _AutoEpochDataLoader  # noqa: E402


def _two_rank_prepare(monkeypatch, loader, **kwargs):
    """Run prepare() as rank 0 of 2 without a real process group or DDP wrap.

    Only the dataloader's journey through prepare() is under test here; the
    collective machinery has its own suite in test_distributed.py.
    """
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch_backend, "_ensure_process_group", lambda: True)
    monkeypatch.setattr("torch.nn.parallel.DistributedDataParallel", lambda m, **kw: m)
    return torch_backend.prepare(nn.Linear(3, 1), loader, auto_launch=False, **kwargs)


def _bare_loader(n=16, batch_size=4, shuffle=False):
    """num_workers=0 / pin_memory=False - what build_loader_defaults acts on.

    ``shuffle`` matters for the ordering tests only: ``_shard_loader`` honors
    the user's choice, so a loader built without it gets a
    ``DistributedSampler(shuffle=False)`` whose ``set_epoch`` is a no-op by
    design. The wrapper is still installed either way.
    """
    return DataLoader(
        TensorDataset(torch.randn(n, 3), torch.randn(n, 1)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


class TestWrapperSurvivesPrepare:
    def test_wrapper_survives_the_optimize_rebuild(self, monkeypatch, pretend_cuda):
        """The regression: optimize=True rebuilt the loader and dropped it."""
        _, loader = _two_rank_prepare(monkeypatch, _bare_loader(), optimize=True)
        assert isinstance(loader, _AutoEpochDataLoader)

    def test_wrapper_present_without_optimize(self, monkeypatch, pretend_cuda):
        """Baseline: with no rebuild to survive, it was always there."""
        _, loader = _two_rank_prepare(monkeypatch, _bare_loader(), optimize=False)
        assert isinstance(loader, _AutoEpochDataLoader)

    def test_iterating_advances_the_sampler_epoch(self, monkeypatch, pretend_cuda):
        """The behavior the wrapper exists for, through the optimize path:
        each pass over the loader bumps set_epoch, so the shuffle changes."""
        _, loader = _two_rank_prepare(monkeypatch, _bare_loader(), optimize=True)

        seen = []
        real_set_epoch = loader.sampler.set_epoch

        def spy(epoch):
            seen.append(epoch)
            return real_set_epoch(epoch)

        monkeypatch.setattr(loader.sampler, "set_epoch", spy)
        for _ in range(3):
            for _batch in loader:
                pass

        assert seen == [0, 1, 2], f"set_epoch not advanced per epoch: {seen}"

    def test_epochs_draw_different_orders(self, monkeypatch, pretend_cuda):
        """End-to-end consequence: two passes must not be the same permutation.

        This is what silently regressed - the run kept training, just on one
        fixed order forever.
        """
        _, loader = _two_rank_prepare(monkeypatch, _bare_loader(n=64, shuffle=True), optimize=True)
        first = [tuple(x.flatten().tolist()) for xb, _ in loader for x in xb]
        second = [tuple(x.flatten().tolist()) for xb, _ in loader for x in xb]
        assert first != second, "every epoch drew the identical permutation"

    def test_shuffle_false_is_still_deterministic(self, monkeypatch, pretend_cuda):
        """The wrapper must not smuggle shuffling into a loader that asked for
        none - set_epoch is installed, but shuffle=False keeps order fixed."""
        _, loader = _two_rank_prepare(monkeypatch, _bare_loader(n=64, shuffle=False), optimize=True)
        assert loader.sampler.shuffle is False
        first = [tuple(x.flatten().tolist()) for xb, _ in loader for x in xb]
        second = [tuple(x.flatten().tolist()) for xb, _ in loader for x in xb]
        assert first == second

    def test_loader_optimizations_still_applied(self, monkeypatch, pretend_cuda):
        """Re-wrapping must not cost the optimize win it wraps: the rebuilt
        loader's kwargs have to show through the wrapper's __getattr__."""
        _, loader = _two_rank_prepare(monkeypatch, _bare_loader(), optimize=True)
        assert isinstance(loader, _AutoEpochDataLoader)
        assert loader.num_workers > 0
        assert loader.pin_memory is True

    def test_user_installed_sampler_is_left_unwrapped(self, monkeypatch, pretend_cuda):
        """A user who sharded the loader themselves owns set_epoch; prepare()
        passes their loader through and must not start managing it."""
        from torch.utils.data.distributed import DistributedSampler

        dataset = TensorDataset(torch.randn(16, 3), torch.randn(16, 1))
        sampler = DistributedSampler(dataset, num_replicas=2, rank=0)
        user_loader = DataLoader(dataset, batch_size=4, sampler=sampler)

        _, loader = _two_rank_prepare(monkeypatch, user_loader, optimize=False)
        assert not isinstance(loader, _AutoEpochDataLoader)


class TestSingleProcessUnaffected:
    def test_no_wrapper_when_not_distributed(self, monkeypatch, pretend_cuda):
        """world_size == 1 shards nothing, so there is no sampler to advance."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        _, loader = torch_backend.prepare(
            nn.Linear(3, 1), _bare_loader(), optimize=True, auto_launch=False
        )
        assert not isinstance(loader, _AutoEpochDataLoader)
