"""Gradient boosting backend (XGBoost, LightGBM).

Tree libraries have no batch size or gradient sync; parallelism means
threads for tree construction. v0.4a: single-node thread auto-tuning,
reusing the SLURM-aware CPU detection from the sklearn backend.

Multi-node boosting (xgboost.dask / lightgbm.dask on a cluster built from
the SLURM allocation) is planned as v0.4b.

Handles both API styles:
    sklearn API   : XGBClassifier(), LGBMRegressor(), ...  -> set n_jobs
    native params : {"max_depth": 6, ...}                  -> set nthread/num_threads
"""

from __future__ import annotations

import os
from typing import Any

from .sklearn_backend import _available_cpus


def prepare(model: Any, n_jobs: int | None = None) -> Any:
    """Configure thread count on an XGBoost/LightGBM estimator (in place)."""
    jobs = n_jobs if n_jobs is not None else _available_cpus()
    lib = type(model).__module__.split(".")[0]

    if hasattr(model, "set_params"):
        model.set_params(n_jobs=jobs)
    else:
        raise TypeError(
            f"Expected a scikit-learn-API estimator, got {type(model)!r}. "
            "For native training, use autotrainer.boost_params() instead."
        )

    _warn_if_multinode()
    print(
        f"[autotrainer] {lib} backend: n_jobs={jobs} "
        f"(source={'SLURM' if os.environ.get('SLURM_CPUS_PER_TASK') else 'local cores'})"
    )
    return model


def boost_params(params: dict | None = None, lib: str = "xgboost") -> dict:
    """Return a params dict with the right thread key set, for native APIs.

    Example:
        params = autotrainer.boost_params({"max_depth": 6})
        xgboost.train(params, dtrain)
    """
    jobs = _available_cpus()
    out = dict(params or {})
    out["nthread" if lib == "xgboost" else "num_threads"] = jobs
    _warn_if_multinode()
    return out


def _warn_if_multinode() -> None:
    if int(os.environ.get("SLURM_NNODES", "1")) > 1:
        print(
            "[autotrainer] note: multi-node boosting (xgboost.dask) lands in "
            "v0.4b - currently using threads on this node only."
        )
