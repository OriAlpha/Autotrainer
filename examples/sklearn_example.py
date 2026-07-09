"""scikit-learn: auto n_jobs (respects SLURM_CPUS_PER_TASK on clusters).
Run: python sklearn_example.py
"""
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import autotrainer

X, y = make_classification(n_samples=5000, n_features=20)

# Works on plain estimators and on nested ones (pipeline inside grid search)
pipe = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier())])
search = GridSearchCV(pipe, {"rf__n_estimators": [100, 200], "rf__max_depth": [None, 10]})
search = autotrainer.prepare(search)            # n_jobs set everywhere it applies

search.fit(X, y)
print("best score:", search.best_score_)
