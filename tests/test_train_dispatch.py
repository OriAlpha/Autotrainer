"""Framework routing for ``autotrainer.train()``.

``train()`` used to pick a backend by probing attributes -
``isinstance(model, dict)`` for native XGBoost, then
``hasattr(model, "fit") and not hasattr(model, "forward")`` for
sklearn/Keras. Anything else fell through to the PyTorch path and died deep
inside it on ``next(model.parameters())``.

It now classifies with ``utils.framework_of``, the same module-prefix rule
``prepare()`` and ``tune()`` use, so every entry point agrees on what a model
is. These tests pin the routing decision itself; the per-backend training
behavior is covered by the backend suites.
"""

from __future__ import annotations

import pytest

from autotrainer.utils import framework_of


class TestFrameworkOf:
    """The shared classifier. Order matters: boosting must be decided before
    sklearn, because XGBoost/LightGBM's sklearn-API classes subclass
    ``BaseEstimator`` and would otherwise be misrouted to joblib's n_jobs
    handling instead of the boosting backend's thread config."""

    def test_torch_module(self):
        torch = pytest.importorskip("torch")
        assert framework_of(torch.nn.Linear(2, 1)) == "torch"

    def test_torch_subclass_defined_in_user_code(self):
        """A user's nn.Module subclass has __module__ == the test module, so
        this only works via the isinstance fallback."""
        torch = pytest.importorskip("torch")

        class MyNet(torch.nn.Module):
            def forward(self, x):
                return x

        assert framework_of(MyNet()) == "torch"

    def test_sklearn_estimator(self):
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LinearRegression

        assert framework_of(LinearRegression()) == "sklearn"

    def test_xgboost_is_boosting_not_sklearn(self):
        xgb = pytest.importorskip("xgboost")
        from sklearn.base import BaseEstimator

        model = xgb.XGBRegressor()
        # It really is a BaseEstimator - the ordering is what saves us.
        assert isinstance(model, BaseEstimator)
        assert framework_of(model) == "boosting"

    def test_lightgbm_is_boosting_not_sklearn(self):
        lgb = pytest.importorskip("lightgbm")
        assert framework_of(lgb.LGBMRegressor()) == "boosting"

    def test_unknown_object(self):
        class NotAModel:
            pass

        assert framework_of(NotAModel()) == "unknown"

    def test_object_with_a_fit_method_is_not_assumed_sklearn(self):
        """The old duck-typed check routed anything with .fit() and no
        .forward() straight to the sklearn backend."""

        class HasFit:
            def fit(self, *a, **kw):
                raise AssertionError("should never be called")

        assert framework_of(HasFit()) == "unknown"


class TestTrainRejectsUnroutableModels:
    def test_unknown_type_raises_typeerror(self):
        pytest.importorskip("torch")
        import autotrainer

        class NotAModel:
            pass

        with pytest.raises(TypeError, match="train\\(\\) supports"):
            autotrainer.train(NotAModel())

    def test_object_with_fit_no_longer_silently_uses_sklearn(self):
        pytest.importorskip("torch")
        import autotrainer

        class HasFit:
            def fit(self, *a, **kw):
                raise AssertionError("should never be called")

        with pytest.raises(TypeError, match="train\\(\\) supports"):
            autotrainer.train(HasFit())

    def test_params_dict_without_dmatrix_raises_clearly(self):
        pytest.importorskip("torch")
        import autotrainer

        with pytest.raises(TypeError, match="native-XGBoost path"):
            autotrainer.train({"max_depth": 6})
