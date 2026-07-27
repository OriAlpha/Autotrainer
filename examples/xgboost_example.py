"""XGBoost & LightGBM: auto thread count for both API styles.
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

    # sklearn-style API
    clf = autotrainer.prepare(XGBClassifier(n_estimators=100))
    clf.fit(X, y)
    print("xgb sklearn-API accuracy:", clf.score(X, y))

    # native API: get a params dict with nthread already set
    params = autotrainer.boost_params({"max_depth": 6, "objective": "binary:logistic"})
    booster = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=50)
    print("xgb native API trained,", booster.num_boosted_rounds(), "rounds")


if __name__ == "__main__":
    main()
