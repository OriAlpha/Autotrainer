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

The tests are split by what they need, because the obvious single-suite
version is a trap: ``pretend_cuda`` makes ``prepare()`` take the CUDA path and
set ``pin_memory=True`` / ``num_workers>0``, and *iterating* such a loader
sends torch's pin thread into a real driver that CPU-only CI does not have.
So:

* :class:`TestAutoEpochWrapper` drives the wrapper directly on CPU - its
  contract (advance the epoch, change the order) needs no device.
* :class:`TestWrapperSurvivesPrepare` uses ``pretend_cuda`` to reach the
  rebuild that caused the bug, and only *inspects* the result.
* :class:`TestAutoEpochOnRealGpu` does the end-to-end iteration, gated on the
  ``cuda`` marker so it runs on the GPU runner and skips elsewhere.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from torch.utils.data.distributed import DistributedSampler  # noqa: E402

from autotrainer.backends import torch_backend  # noqa: E402
from autotrainer.backends.torch_backend import _AutoEpochDataLoader  # noqa: E402

_HAS_CUDA = torch.cuda.is_available()


def _dataset(n=32):
    return TensorDataset(torch.randn(n, 3), torch.randn(n, 1))


def _bare_loader(n=16, batch_size=4, shuffle=False):
    """num_workers=0 / pin_memory=False - what build_loader_defaults acts on."""
    return DataLoader(_dataset(n), batch_size=batch_size, shuffle=shuffle)


def _order(loader):
    """Flattened sample order of one full pass, for comparing permutations."""
    return [tuple(x.flatten().tolist()) for xb, _ in loader for x in xb]


class TestAutoEpochWrapper:
    """The wrapper's own contract. Pure CPU: no device is involved in
    advancing a sampler's epoch counter."""

    def _wrapped(self, n=32, shuffle=True):
        sampler = DistributedSampler(_dataset(n), num_replicas=2, rank=0, shuffle=shuffle)
        return _AutoEpochDataLoader(DataLoader(_dataset(n), batch_size=4, sampler=sampler))

    def test_each_pass_advances_the_epoch(self):
        loader = self._wrapped()
        seen = []
        real = loader.sampler.set_epoch
        loader.sampler.set_epoch = lambda e: (seen.append(e), real(e))[1]

        for _ in range(3):
            for _batch in loader:
                pass

        assert seen == [0, 1, 2], f"set_epoch not advanced per pass: {seen}"

    def test_shuffled_passes_draw_different_orders(self):
        """The consequence that silently regressed - the run kept training,
        just on one fixed order forever."""
        loader = self._wrapped(shuffle=True)
        assert _order(loader) != _order(loader)

    def test_unshuffled_passes_stay_deterministic(self):
        """The wrapper must not smuggle shuffling into a loader that asked for
        none: set_epoch still advances, but shuffle=False fixes the order."""
        loader = self._wrapped(shuffle=False)
        assert _order(loader) == _order(loader)

    def test_delegates_len_and_attributes(self):
        """Callers treat the wrapper as the DataLoader it stands in for."""
        loader = self._wrapped()
        assert len(loader) == len(loader.dataloader)
        assert loader.batch_size == 4
        assert loader.num_workers == 0


class TestWrapperSurvivesPrepare:
    """The regression itself: the wrapper must outlive prepare()'s rebuilds.

    These only inspect the returned loader. Iterating one built under
    ``pretend_cuda`` would touch a real driver - see the module docstring.
    """

    def _two_rank_prepare(self, monkeypatch, loader, **kwargs):
        """Run prepare() as rank 0 of 2 without a real process group or DDP
        wrap. Only the dataloader's journey is under test here; the collective
        machinery has its own suite in test_distributed.py.
        """
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setattr(torch_backend, "_ensure_process_group", lambda: True)
        monkeypatch.setattr("torch.nn.parallel.DistributedDataParallel", lambda m, **kw: m)
        return torch_backend.prepare(nn.Linear(3, 1), loader, auto_launch=False, **kwargs)

    def test_wrapper_survives_the_optimize_rebuild(self, monkeypatch, pretend_cuda):
        """The bug: optimize=True rebuilt the loader and dropped the wrapper,
        so set_epoch was never called again."""
        _, loader = self._two_rank_prepare(monkeypatch, _bare_loader(), optimize=True)
        assert isinstance(loader, _AutoEpochDataLoader)

    def test_wrapper_present_without_optimize(self, monkeypatch, pretend_cuda):
        """Baseline: with no rebuild to survive, it was always there."""
        _, loader = self._two_rank_prepare(monkeypatch, _bare_loader(), optimize=False)
        assert isinstance(loader, _AutoEpochDataLoader)

    def test_loader_optimizations_still_applied(self, monkeypatch, pretend_cuda):
        """Re-wrapping must not cost the optimize win it wraps: the rebuilt
        loader's kwargs have to show through the wrapper's __getattr__."""
        _, loader = self._two_rank_prepare(monkeypatch, _bare_loader(), optimize=True)
        assert isinstance(loader, _AutoEpochDataLoader)
        assert loader.num_workers > 0
        assert loader.pin_memory is True

    def test_sampler_is_distributed_and_shuffle_choice_honored(self, monkeypatch, pretend_cuda):
        _, loader = self._two_rank_prepare(monkeypatch, _bare_loader(shuffle=True), optimize=True)
        assert isinstance(loader.sampler, DistributedSampler)
        assert loader.sampler.shuffle is True

    def test_user_installed_sampler_is_left_unwrapped(self, monkeypatch, pretend_cuda):
        """A user who sharded the loader themselves owns set_epoch; prepare()
        passes their loader through and must not start managing it."""
        dataset = _dataset(16)
        sampler = DistributedSampler(dataset, num_replicas=2, rank=0)
        user_loader = DataLoader(dataset, batch_size=4, sampler=sampler)

        _, loader = self._two_rank_prepare(monkeypatch, user_loader, optimize=False)
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


@pytest.mark.cuda
@pytest.mark.skipif(not _HAS_CUDA, reason="iterates a pin_memory loader, needs a real GPU")
class TestAutoEpochOnRealGpu:
    """End-to-end on a real device: prepare()'s loader, actually iterated.

    The classes above split the proof in two - the wrapper reshuffles, and
    prepare() keeps the wrapper - which is only a proof of the whole if the
    two halves meet. They do meet here, on hardware where iterating a
    pin_memory loader is legal.
    """

    def test_iterating_the_prepared_loader_advances_epochs(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setattr(torch_backend, "_ensure_process_group", lambda: True)
        monkeypatch.setattr("torch.nn.parallel.DistributedDataParallel", lambda m, **kw: m)

        _, loader = torch_backend.prepare(
            nn.Linear(3, 1),
            _bare_loader(n=64, shuffle=True),
            optimize=True,
            auto_launch=False,
        )
        assert isinstance(loader, _AutoEpochDataLoader)

        seen = []
        real = loader.sampler.set_epoch
        loader.sampler.set_epoch = lambda e: (seen.append(e), real(e))[1]

        first = _order(loader)
        second = _order(loader)

        assert seen == [0, 1]
        assert first != second, "every epoch drew the identical permutation"
