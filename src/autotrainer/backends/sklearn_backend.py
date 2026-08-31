"""scikit-learn backend.

Classical ML doesn't do gradient synchronization; parallelism here means
joblib workers. The right worker count depends on the environment:

- SLURM job:  use SLURM_CPUS_PER_TASK (respect the allocation, never
              oversubscribe a shared node)
- local:      use the affinity mask (respects cgroups/containers)

Both come from :func:`autotrainer.utils.available_cpus`, shared with the
DataLoader worker sizing so the two never disagree about the allocation.

`prepare()` sets `n_jobs` on the estimator and, recursively, on any nested
estimators (pipelines, ensembles, CV wrappers) that accept it.
"""

from __future__ import annotations

import os
from typing import Any

from ..utils import available_cpus


def prepare(model: Any, n_jobs: int | None = None) -> Any:
    """Set n_jobs on the estimator and any nested estimators that support it.

    Returns the same estimator, configured in place.
    """
    jobs = n_jobs if n_jobs is not None else available_cpus()

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

    from ..summary import get_active_summary

    get_active_summary()

    return model
