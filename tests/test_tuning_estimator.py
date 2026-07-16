"""Tests for tuning_estimator.py: Optuna search for sklearn-API models."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("optuna")
import numpy as np  # noqa: E402

import autotrainer  # noqa: E402
from autotrainer.tuning_estimator import _default_space, tune_estimator  # noqa: E402


def _classification_arrays(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    cut = int(n * 0.75)
    return (x[:cut], y[:cut]), (x[cut:], y[cut:])


class TestDefaultSpace:
    def test_known_families_get_a_space(self):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression, Ridge

        for est in (
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            LogisticRegression(),
            Ridge(),
        ):
            assert _default_space(est), f"no default space for {type(est).__name__}"

    def test_xgboost_gets_the_xgb_space(self):
        xgboost = pytest.importorskip("xgboost")
        from autotrainer.tuning_estimator import _XGB_SPACE

        assert _default_space(xgboost.XGBClassifier()) is _XGB_SPACE

    def test_unknown_estimator_has_no_space(self):
        from sklearn.cluster import KMeans

        assert _default_space(KMeans()) is None


class TestTuneEstimator:
    def test_searches_refits_and_never_touches_input(self):
        from sklearn.exceptions import NotFittedError
        from sklearn.linear_model import LogisticRegression
        from sklearn.utils.validation import check_is_fitted

        train, val = _classification_arrays()
        model = LogisticRegression(max_iter=200)
        best, params, study = autotrainer.tune(model, train, val, trials=3, verbose=False)

        assert set(params) == {"C"}
        assert len(study.trials) == 3
        assert best.score(*val) > 0.5  # fitted and sane on an easy problem
        with pytest.raises(NotFittedError):
            check_is_fitted(model)  # the user's estimator stays untouched

    def test_unknown_estimator_without_space_raises(self):
        from sklearn.cluster import KMeans

        train, val = _classification_arrays()
        with pytest.raises(TypeError, match="No default search space"):
            tune_estimator(KMeans(), train, val, trials=1, verbose=False)

    def test_loader_style_input_raises_clear_error(self):
        from sklearn.linear_model import LogisticRegression

        with pytest.raises(TypeError, match=r"\(X, y\) tuple"):
            tune_estimator(LogisticRegression(), "not-a-tuple", "nope", trials=1)

    def test_custom_scoring_is_used(self):
        from sklearn.linear_model import LogisticRegression

        train, val = _classification_arrays()
        seen = []

        def scoring(est, x, y):
            seen.append(1)
            return float(est.score(x, y))

        tune_estimator(
            LogisticRegression(max_iter=200), train, val, trials=2, scoring=scoring, verbose=False
        )
        assert len(seen) >= 2  # once per trial (+ nothing for the refit)

    def test_xgboost_end_to_end_via_dispatcher(self):
        xgboost = pytest.importorskip("xgboost")

        train, val = _classification_arrays()
        small_space = {"n_estimators": ("int", 10, 20), "max_depth": ("int", 2, 4)}
        best, params, study = autotrainer.tune(
            xgboost.XGBClassifier(), train, val, trials=2, space=small_space, verbose=False
        )
        assert set(params) == {"n_estimators", "max_depth"}
        assert best.score(*val) > 0.5

    def test_fit_rejects_estimators_with_helpful_error(self):
        from sklearn.linear_model import LogisticRegression

        train, val = _classification_arrays()
        with pytest.raises(TypeError, match="autotrainer.tune"):
            autotrainer.fit(LogisticRegression(), train, val)
