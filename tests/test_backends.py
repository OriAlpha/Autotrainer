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
        # Model moved to CUDA when available, stays on CPU otherwise.
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert next(out.parameters()).device.type == expected


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


class TestTorchPrepareWarnings:
    def test_single_process_fsdp_and_offload_warns(self, capsys):
        pytest.importorskip("torch")
        import torch.nn as nn

        model = nn.Linear(4, 2)
        # 1. fsdp on world_size == 1
        prepare(model, fsdp=True)
        out = capsys.readouterr().out
        assert "world_size == 1, FSDP is a no-op" in out

        # 2. fsdp with cpu_offload on world_size == 1
        prepare(model, fsdp=True, cpu_offload=True)
        out = capsys.readouterr().out
        assert "cpu_offload: ignored (world_size == 1)" in out

        # 3. cpu_offload without fsdp
        prepare(model, fsdp=False, cpu_offload=True)
        out = capsys.readouterr().out
        assert "cpu_offload: ignored (world_size == 1 and fsdp=False" in out

        # 4. static_graph on single process
        prepare(model, static_graph=True)
        out = capsys.readouterr().out
        assert "static_graph: ignored (world_size == 1" in out

    def test_compile_fallback_warns_on_failure(self, monkeypatch, capsys):
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        model = nn.Linear(4, 2)

        def mock_compile(m, *a, **kw):
            raise RuntimeError("Inductor backend failure simulation")

        monkeypatch.setattr(torch, "compile", mock_compile)
        res = prepare(model, compile=True)
        out = capsys.readouterr().out
        assert "compile failed" in out
        assert "continuing with the uncompiled model" in out
        assert res is model


class TestTFBackend:
    def test_slurm_hostnames_resolution(self, monkeypatch):
        from autotrainer.backends.tf_backend import _slurm_hostnames

        # 1. With scontrol
        monkeypatch.setattr(
            "shutil.which", lambda cmd: "/usr/bin/scontrol" if cmd == "scontrol" else None
        )
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **kw: type("Res", (), {"stdout": "worker1\nworker2\n"})()
        )
        monkeypatch.setenv("SLURM_NODELIST", "worker[1-2]")
        assert _slurm_hostnames() == ["worker1", "worker2"]

        # 2. Fallback without scontrol
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        monkeypatch.setenv("SLURM_NODELIST", "node1,node2")
        assert _slurm_hostnames() == ["node1", "node2"]

    def test_scope_single_device(self, capsys):
        pytest.importorskip("tensorflow")
        from autotrainer.backends.tf_backend import scope

        with scope():
            pass
        out = capsys.readouterr().out
        assert "tf backend: default strategy" in out

    def test_scope_slurm_multiworker(self, monkeypatch, capsys):
        tf = pytest.importorskip("tensorflow")
        from autotrainer.backends.tf_backend import scope

        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_NNODES", "2")
        monkeypatch.setenv("SLURM_NODELIST", "node01,node02")
        monkeypatch.setenv("SLURM_NODEID", "0")

        # Mock MultiWorkerMirroredStrategy to avoid real cluster networking in tests
        class MockStrategy:
            num_replicas_in_sync = 2

            def scope(self):
                return tf.distribute.get_strategy().scope()

        monkeypatch.setattr(tf.distribute, "MultiWorkerMirroredStrategy", MockStrategy)
        with scope():
            pass
        out = capsys.readouterr().out
        assert "MultiWorkerMirroredStrategy (2 nodes, 2 replicas)" in out

    def test_scope_mirrored_multi_gpu(self, monkeypatch, capsys):
        tf = pytest.importorskip("tensorflow")
        from autotrainer.backends.tf_backend import scope

        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setattr(
            tf.config, "list_physical_devices", lambda dev: [1, 2] if dev == "GPU" else []
        )

        class MockMirroredStrategy:
            def scope(self):
                return tf.distribute.get_strategy().scope()

        monkeypatch.setattr(tf.distribute, "MirroredStrategy", MockMirroredStrategy)
        with scope():
            pass
        out = capsys.readouterr().out
        assert "MirroredStrategy (2 GPUs)" in out
