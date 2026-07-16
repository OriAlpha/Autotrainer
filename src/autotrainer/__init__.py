"""autotrainer: automatic distributed training.

Usage inside a training script (one line before the loop):

    import autotrainer
    model, loader, opt = autotrainer.prepare(model, loader, opt)

Launch with:

    autotrainer run train.py            # local: auto-detects GPUs
    srun autotrainer run train.py       # inside an sbatch script on SLURM
"""

from __future__ import annotations

from typing import Any

__version__ = "0.10.0"

# The public API. Everything not listed here (submodules, _-prefixed
# helpers) is internal and may change without a deprecation cycle.
__all__ = [
    "GradScaler",
    "__version__",
    "auto",
    "autocast_context",
    "barrier",
    "boost_params",
    "find_batch_size",
    "find_lr",
    "fit",
    "is_main",
    "prepare",
    "print0",
    "rank",
    "save0",
    "scale_batch_size",
    "scope",
    "set_epoch",
    "tune",
]

from .utils import (  # noqa: E402,F401
    GradScaler,
    autocast_context,
    barrier,
    is_main,
    print0,
    rank,
    save0,
    set_epoch,
)


def scope() -> Any:
    """TensorFlow: strategy scope. Create and compile your model inside it."""
    from .backends.tf_backend import scope as _s

    return _s()


def scale_batch_size(per_replica_batch: int) -> int:
    """TensorFlow: convert a per-replica batch size to the global batch size."""
    from .backends.tf_backend import scale_batch_size as _s

    return _s(per_replica_batch)


def boost_params(params: dict[str, Any] | None = None, lib: str = "xgboost") -> dict[str, Any]:
    """XGBoost/LightGBM native API: params dict with auto thread count."""
    from .backends.boosting_backend import boost_params as _b

    return _b(params, lib=lib)


def prepare(model: Any, dataloader: Any = None, optimizer: Any = None) -> Any:
    """Framework dispatcher: route to the right backend by model type."""
    mod = type(model).__module__ or ""

    if mod.startswith("torch") or _is_torch_module(model):
        from .backends.torch_backend import prepare as torch_prepare

        return torch_prepare(model, dataloader, optimizer)

    if mod.startswith(("keras", "tensorflow")):
        raise TypeError(
            "TensorFlow models must be created inside `with autotrainer.scope():` "
            "rather than passed to prepare() - see README."
        )

    # Must come before the sklearn check: XGB/LGBM sklearn-API models
    # subclass BaseEstimator and would be misrouted otherwise.
    if mod.startswith(("xgboost", "lightgbm")):
        from .backends.boosting_backend import prepare as boost_prepare

        return boost_prepare(model)

    if mod.startswith("sklearn") or _is_sklearn_estimator(model):
        from .backends.sklearn_backend import prepare as sklearn_prepare

        return sklearn_prepare(model)

    raise TypeError(f"Unrecognized model type: {type(model)!r}")


def find_batch_size(model: Any, sample_batch_fn: Any, start: int = 2, max_bs: int = 4096) -> int:
    from .backends.torch_backend import find_batch_size as _f

    return _f(model, sample_batch_fn, start=start, max_bs=max_bs)


def auto(model: Any, dataloader: Any, **kwargs: Any) -> tuple[Any, ...]:
    """PyTorch: infer loss, optimizer, LR, and schedule, then distribute."""
    from .auto_optim import auto as _a

    return _a(model, dataloader, **kwargs)


def find_lr(model: Any, dataloader: Any, loss_fn: Any, **kwargs: Any) -> float:
    """PyTorch: LR range test on a throwaway model copy."""
    from .auto_optim import find_lr as _f

    return _f(model, dataloader, loss_fn, **kwargs)


def tune(model: Any, train: Any = None, val: Any = None, **kwargs: Any) -> tuple[Any, ...]:
    """Search hyperparameters for a model (dispatches by framework).

    PyTorch modules: pass DataLoaders; searches the training recipe
    (lr, weight decay, optimizer, batch size).
    sklearn-API estimators (scikit-learn, XGBoost, LightGBM): pass
    ``(X, y)`` tuples; searches the model's hyperparameters, with curated
    default spaces for the common families.

    The keyword names ``train_loader=``/``val_loader=`` are deprecated
    aliases for ``train=``/``val=`` and will be removed in 1.0.
    """
    # Deprecated aliases (pre-0.10 names; misleading for estimator inputs).
    for old, new, value in (("train_loader", "train", train), ("val_loader", "val", val)):
        if old in kwargs:
            if value is not None:
                raise TypeError(f"tune() got both '{new}' and deprecated '{old}'")
            import warnings

            warnings.warn(
                f"tune(..., {old}=) is deprecated; use {new}= (removal in 1.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            if old == "train_loader":
                train = kwargs.pop(old)
            else:
                val = kwargs.pop(old)
    if train is None or val is None:
        raise TypeError("tune() requires train and val data")

    mod = type(model).__module__ or ""
    if mod.startswith("torch") or _is_torch_module(model):
        from .tuning import tune as _t

        return _t(model, train, val, **kwargs)
    if mod.startswith(("xgboost", "lightgbm", "sklearn")) or _is_sklearn_estimator(model):
        from .tuning_estimator import tune_estimator as _te

        return _te(model, train, val, **kwargs)
    raise TypeError(
        f"tune() supports PyTorch modules and sklearn-API estimators, got {type(model)!r}."
    )


def fit(model: Any, train_loader: Any, val_loader: Any, **kwargs: Any) -> tuple[Any, ...]:
    """PyTorch: tune the recipe, then fully train the winner (with early stopping)."""
    from .fitting import fit as _f

    return _f(model, train_loader, val_loader, **kwargs)


def _is_torch_module(model: Any) -> bool:
    try:
        import torch  # noqa: PLC0415

        return isinstance(model, torch.nn.Module)
    except ImportError:
        return False


def _is_sklearn_estimator(model: Any) -> bool:
    try:
        from sklearn.base import BaseEstimator  # noqa: PLC0415

        return isinstance(model, BaseEstimator)
    except ImportError:
        return False
