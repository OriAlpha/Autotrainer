"""scikit-learn backend.

Classical ML doesn't do gradient synchronization; parallelism here means
joblib workers. The right worker count depends on the environment:

- SLURM job:  use SLURM_CPUS_PER_TASK (respect the allocation, never
              oversubscribe a shared node)
- local:      use physical core count

`prepare()` sets `n_jobs` on the estimator and, recursively, on any nested
estimators (pipelines, ensembles, CV wrappers) that accept it.
"""

from __future__ import annotations

import os
from typing import Any


def _available_cpus() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(int(slurm_cpus), 1)
    # sched_getaffinity respects cgroups/containers, unlike cpu_count. It is
    # POSIX-only, so guard with hasattr instead of relying on a type: ignore
    # (which mypy would flag as unused on Linux where the attr IS defined).
    if hasattr(os, "sched_getaffinity"):
        return max(len(os.sched_getaffinity(0)), 1)
    return max(os.cpu_count() or 1, 1)


def prepare(model: Any, n_jobs: int | None = None) -> Any:
    """Set n_jobs on the estimator and any nested estimators that support it.

    Returns the same estimator, configured in place.
    """
    jobs = n_jobs if n_jobs is not None else _available_cpus()

    params = model.get_params(deep=True)
    updates = {k: jobs for k in params if k == "n_jobs" or k.endswith("__n_jobs")}
    if updates:
        model.set_params(**updates)

    verb = "set" if updates else "no n_jobs parameter found on"
    print(
        f"[autotrainer] sklearn backend: {verb} {type(model).__name__} "
        f"(workers={jobs}, source="
        f"{'SLURM' if os.environ.get('SLURM_CPUS_PER_TASK') else 'local cores'})"
    )
    return model
