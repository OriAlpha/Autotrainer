"""XGBoost & LightGBM: auto thread count for both API styles.
Run: python xgboost_example.py
"""
from sklearn.datasets import make_classification

import autotrainer

X, y = make_classification(n_samples=5000, n_features=20)

# sklearn-style API
from xgboost import XGBClassifier
clf = autotrainer.prepare(XGBClassifier(n_estimators=100))
clf.fit(X, y)
print("xgb sklearn-API accuracy:", clf.score(X, y))

# native API: get a params dict with nthread already set
import xgboost as xgb
params = autotrainer.boost_params({"max_depth": 6, "objective": "binary:logistic"})
booster = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=50)
print("xgb native API trained,", booster.num_boosted_rounds(), "rounds")
