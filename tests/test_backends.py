"""Tests for the per-framework backends (torch / sklearn / boosting)."""

from __future__ import annotations

import pytest

from autotrainer.backends.torch_backend import _dist_info, find_batch_size, prepare


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
