"""Tests for the per-framework backends (torch / sklearn / boosting)."""

from __future__ import annotations

import pytest

from autotrainer.backends.torch_backend import (
    _dist_info,
    _loader_kwargs,
    _shard_loader,
    find_batch_size,
    prepare,
)


class TestTorchDistInfo:
    def test_reads_rank_vars_from_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert _dist_info() == (3, 1, 8)

    def test_defaults_when_unset(self):
        # clean_env fixture has already stripped the vars.
        assert _dist_info() == (0, 0, 1)


class TestTorchPrepareSingleDevice:
    def test_single_device_does_not_init_process_group(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import torch.distributed as dist

        monkeypatch.setenv("WORLD_SIZE", "1")
        model = torch.nn.Linear(3, 2)
        out = prepare(model)
        # Single device: no DDP wrap, no process group.
        assert not isinstance(out, torch.nn.parallel.DistributedDataParallel)
        assert not dist.is_initialized()
        # Model should be on CPU (no CUDA in CI).
        assert next(out.parameters()).device.type == "cpu"


class TestShardLoader:
    def _dataset(self, n=16):
        torch = pytest.importorskip("torch")
        from torch.utils.data import TensorDataset

        return TensorDataset(torch.randn(n, 3), torch.randint(0, 2, (n,)))

    def test_preserves_user_loader_settings(self):
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        def collate(batch):
            return torch.utils.data.default_collate(batch)

        def init_fn(worker_id):
            pass

        gen = torch.Generator()
        loader = DataLoader(
            self._dataset(),
            batch_size=4,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate,
            worker_init_fn=init_fn,
            generator=gen,
            persistent_workers=True,
            prefetch_factor=4,
        )
        out = _shard_loader(loader, rank=0, world_size=2)
        assert isinstance(out.sampler, DistributedSampler)
        assert out.sampler.shuffle  # user had shuffle=True
        assert out.batch_size == 4
        assert out.num_workers == 2
        assert out.pin_memory is True
        assert out.drop_last is True
        assert out.collate_fn is collate
        assert out.worker_init_fn is init_fn
        assert out.generator is gen
        assert out.persistent_workers is True
        assert out.prefetch_factor == 4

    def test_sequential_loader_keeps_shuffle_off(self):
        pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        loader = DataLoader(self._dataset(), batch_size=4)  # shuffle=False
        out = _shard_loader(loader, rank=0, world_size=2)
        assert out.sampler.shuffle is False

    def test_existing_distributed_sampler_passes_through(self):
        pytest.importorskip("torch")
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        ds = self._dataset()
        sampler = DistributedSampler(ds, num_replicas=2, rank=1)
        loader = DataLoader(ds, batch_size=4, sampler=sampler)
        assert _shard_loader(loader, rank=1, world_size=2) is loader

    def test_batch_sampler_loader_raises_clear_error(self):
        pytest.importorskip("torch")
        from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

        ds = self._dataset()
        bs = BatchSampler(SequentialSampler(ds), batch_size=4, drop_last=False)
        loader = DataLoader(ds, batch_sampler=bs)
        with pytest.raises(TypeError, match="batch_sampler"):
            _shard_loader(loader, rank=0, world_size=2)

    def test_iterable_dataset_raises_clear_error(self):
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader, IterableDataset

        class Stream(IterableDataset):
            def __iter__(self):
                return iter([torch.zeros(3)])

        loader = DataLoader(Stream(), batch_size=2)
        with pytest.raises(TypeError, match="IterableDataset"):
            _shard_loader(loader, rank=0, world_size=2)

    def test_loader_kwargs_omits_prefetch_without_workers(self):
        pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        loader = DataLoader(self._dataset(), batch_size=4)  # num_workers=0
        kwargs = _loader_kwargs(loader)
        # DataLoader(prefetch_factor=...) raises when num_workers == 0.
        assert "prefetch_factor" not in kwargs
        DataLoader(loader.dataset, batch_size=4, **kwargs)  # must not raise


class TestFindBatchSize:
    def test_doubles_then_backs_off_on_oom(self):
        pytest.importorskip("torch")

        # Simulated forward+backward: succeeds for bs <= 8, OOMs above that.
        def sample_batch_fn(bs: int) -> None:
            if bs > 8:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2GiB")

        best = find_batch_size(None, sample_batch_fn, start=2, max_bs=64)
        # 2 -> 4 -> 8 succeed; 16 raises OOM; back off to last good = 8.
        assert best == 8

    def test_returns_start_when_immediately_oom(self):
        pytest.importorskip("torch")

        def always_oom(bs: int) -> None:
            raise RuntimeError("cuda out of memory")

        assert find_batch_size(None, always_oom, start=4, max_bs=64) == 4

    def test_non_oom_runtime_error_propagates(self):
        pytest.importorskip("torch")

        def other_error(bs: int) -> None:
            raise RuntimeError("something else broke")

        with pytest.raises(RuntimeError, match="something else"):
            find_batch_size(None, other_error, start=2, max_bs=64)


class TestSklearnBackend:
    def test_prepare_sets_njobs_on_nested_pipeline(self):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        from autotrainer.backends.sklearn_backend import prepare

        pipe = Pipeline([("clf", RandomForestClassifier())])
        out = prepare(pipe, n_jobs=4)
        # The nested estimator should inherit the n_jobs setting.
        assert out.get_params()["clf__n_jobs"] == 4


class TestBoostingBackend:
    def test_prepare_native_object_without_set_params_raises(self):
        pytest.importorskip("xgboost")  # only to keep the extra relevant
        from autotrainer.backends.boosting_backend import prepare

        with pytest.raises(TypeError, match="Expected a scikit-learn-API estimator"):
            prepare(object())  # plain object has no set_params
