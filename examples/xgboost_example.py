"""XGBoost & LightGBM: auto thread count and 1-line execution for both API styles.
Run: python xgboost_example.py

XGBoost/LightGBM parallelize with OpenMP threads (not Python processes), so
the guard isn't strictly required here - it's kept for consistency with the
other examples and so this stays safe if you add a spawn-based step.
"""

import xgboost as xgb
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

import autotrainer


def main() -> None:
    X, y = make_classification(n_samples=5000, n_features=20)

    # 1. Sklearn-style API: 1-line thread config, fit, & save!
    clf = autotrainer.train(XGBClassifier(n_estimators=100), X, y, save_path="xgb_sklearn.json")
    print("xgb sklearn-API accuracy:", clf.score(X, y))

    # 2. Native API: 1-line boost_params injection, train, & save!
    params = {"max_depth": 6, "objective": "binary:logistic"}
    dtrain = xgb.DMatrix(X, label=y)
    booster = autotrainer.train(params, dtrain, epochs=50, save_path="xgb_native.json")
    print("xgb native API trained,", booster.num_boosted_rounds(), "rounds")


if __name__ == "__main__":
    main()
