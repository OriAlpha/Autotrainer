"""Shared pytest fixtures.

`clean_env` strips the distributed/SLURM environment variables between tests so
one test's env mutation can't leak into the next. It is autouse so every test
in every file gets it without importing anything.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip SLURM/rank vars so tests don't leak into each other."""
    for k in list(os.environ):
        if k.startswith(("SLURM_", "RANK", "LOCAL_RANK", "WORLD_SIZE", "AUTOTRAINER_")):
            monkeypatch.delenv(k, raising=False)
