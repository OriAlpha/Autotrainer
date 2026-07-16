"""Hyperparameter tuning for sklearn-API estimators (scikit-learn, XGBoost, LightGBM).

The classical-ML counterpart of tuning.py: the user supplies a built
estimator and arrays, we search its hyperparameters with Optuna.

    best_est, best_params, study = autotrainer.tune(
        XGBClassifier(), (X_train, y_train), (X_val, y_val), trials=50
    )

Known model families get a curated default search space; anything else
needs an explicit ``space=``. Scoring uses the estimator's own ``.score()``
(accuracy for classifiers, R^2 for regressors) unless ``scoring=`` is
given; the direction is always maximize. Thread counts follow the SLURM
allocation via the sklearn backend's CPU detection.
"""

from __future__ import annotations

from typing import Any, Callable

from .tuning import _suggest

_XGB_SPACE = {
    "n_estimators": ("int", 100, 1000),
    "learning_rate": ("loguniform", 1e-3, 0.3),
    "max_depth": ("int", 3, 10),
    "subsample": ("uniform", 0.5, 1.0),
    "colsample_bytree": ("uniform", 0.5, 1.0),
    "min_child_weight": ("loguniform", 1e-1, 1e2),
    "reg_lambda": ("loguniform", 1e-3, 1e2),
}

_LGBM_SPACE = {
    "n_estimators": ("int", 100, 1000),
    "learning_rate": ("loguniform", 1e-3, 0.3),
    "num_leaves": ("int", 15, 255),
    "max_depth": ("int", 3, 12),
    "subsample": ("uniform", 0.5, 1.0),
    "colsample_bytree": ("uniform", 0.5, 1.0),
    "min_child_samples": ("int", 5, 100),
    "reg_lambda": ("loguniform", 1e-3, 1e2),
}

_FOREST_SPACE = {
    "n_estimators": ("int", 100, 500),
    "max_depth": ("int", 3, 30),
    "min_samples_split": ("int", 2, 20),
    "min_samples_leaf": ("int", 1, 10),
    "max_features": ("categorical", ["sqrt", "log2", None]),
}

_GBDT_SPACE = {
    "n_estimators": ("int", 50, 500),
    "learning_rate": ("loguniform", 1e-3, 0.3),
    "max_depth": ("int", 2, 8),
    "subsample": ("uniform", 0.5, 1.0),
}

_SVM_SPACE = {
    "C": ("loguniform", 1e-3, 1e3),
    "gamma": ("loguniform", 1e-5, 1e1),
}


def _default_space(model: Any) -> dict | None:
    """Curated search space for known model families, else None."""
    mod = type(model).__module__ or ""
    name = type(model).__name__
    if mod.startswith("xgboost"):
        return _XGB_SPACE
    if mod.startswith("lightgbm"):
        return _LGBM_SPACE
    if "RandomForest" in name or "ExtraTrees" in name:
        return _FOREST_SPACE
    if name.startswith("GradientBoosting"):
        return _GBDT_SPACE
    if name == "LogisticRegression":
        return {"C": ("loguniform", 1e-4, 1e2)}
    if name in ("Ridge", "Lasso", "ElasticNet"):
        return {"alpha": ("loguniform", 1e-4, 1e2)}
    if name in ("SVC", "SVR"):
        return _SVM_SPACE
    return None


def _wants_n_jobs(model: Any) -> bool:
    """Whether setting n_jobs actually parallelizes this estimator.

    sklearn 1.8+ deprecates the no-op n_jobs on linear models, so only the
    tree/boosting families that truly fan out across cores get it set.
    """
    mod = type(model).__module__ or ""
    name = type(model).__name__
    return mod.startswith(("xgboost", "lightgbm")) or any(
        marker in name for marker in ("Forest", "ExtraTrees", "Bagging")
    )


def _unpack(data: Any, which: str) -> tuple[Any, Any]:
    if isinstance(data, (tuple, list)) and len(data) == 2:
        return data[0], data[1]
    raise TypeError(
        f"For sklearn-API estimators, {which} data must be an (X, y) tuple of "
        f"arrays, got {type(data)!r}. (DataLoaders are for PyTorch models.)"
    )


def tune_estimator(
    model: Any,
    train: Any,
    val: Any,
    *,
    trials: int = 20,
    space: dict | None = None,
    scoring: Callable[[Any, Any, Any], float] | None = None,
    n_jobs: int | None = None,
    seed: int = 0,
    verbose: bool = True,
    storage: Any = None,
    study_name: str | None = None,
) -> tuple[Any, dict, Any]:
    """Search hyperparameters of a sklearn-API estimator.

    Every trial clones the (unfitted) input estimator, so the user's object
    is never touched; the winner is refit on the training data before being
    returned.

    Args:
        model: any estimator following the sklearn get_params/set_params
            contract (scikit-learn, XGBoost/LightGBM sklearn wrappers).
        train: ``(X_train, y_train)`` arrays.
        val: ``(X_val, y_val)`` arrays used to score each trial.
        trials: number of Optuna trials.
        space: search space as in :func:`autotrainer.tune`; if ``None``, a
            curated default is used for known families (XGBoost, LightGBM,
            random forests, gradient boosting, logistic/ridge/lasso, SVM).
        scoring: ``callable(estimator, X_val, y_val) -> float`` (higher is
            better). Defaults to the estimator's own ``.score()``.
        n_jobs: worker/thread count per trial; defaults to the SLURM
            allocation or the physical core count.
        seed: Optuna TPESampler seed.
        verbose: print the final summary.
        storage / study_name: share one study across processes, as in
            :func:`autotrainer.tune`.

    Returns:
        ``(best_estimator, best_params, study)`` with ``best_estimator``
        already fitted on the training data.
    """
    import optuna
    from sklearn.base import clone

    from .backends.sklearn_backend import _available_cpus

    x_train, y_train = _unpack(train, "train")
    x_val, y_val = _unpack(val, "val")

    space = space or _default_space(model)
    if space is None:
        raise TypeError(
            f"No default search space for {type(model).__name__}; pass "
            "space={'param': ('loguniform'|'uniform'|'int'|'categorical', ...)}"
        )
    jobs = n_jobs if n_jobs is not None else _available_cpus()

    def _configure(params: dict) -> Any:
        est = clone(model)
        est.set_params(**params)
        if _wants_n_jobs(est) and "n_jobs" in est.get_params():
            est.set_params(n_jobs=jobs)
        return est

    def _score(est: Any) -> float:
        if scoring is not None:
            return float(scoring(est, x_val, y_val))
        return float(est.score(x_val, y_val))

    def objective(trial: Any) -> float:
        est = _configure(_suggest(trial, space))
        est.fit(x_train, y_train)
        return _score(est)

    prior_verbosity = optuna.logging.get_verbosity()
    try:
        optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            storage=storage,
            study_name=study_name,
            load_if_exists=storage is not None,
        )
        study.optimize(objective, n_trials=trials)
    finally:
        optuna.logging.set_verbosity(prior_verbosity)

    try:
        best_params, best_value = study.best_params, study.best_value
    except ValueError:  # zero-trial share in a parallel search
        best_params, best_value = {}, float("nan")

    best_est = _configure(best_params)
    best_est.fit(x_train, y_train)
    if verbose:
        print(f"[autotrainer] tune: best val score {best_value:.4f} with {best_params}")
    return best_est, best_params, study
