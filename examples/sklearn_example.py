"""scikit-learn: auto n_jobs (respects SLURM_CPUS_PER_TASK on clusters).
Run: python sklearn_example.py

The ``if __name__ == "__main__":`` guard is required on Windows/macOS-spawn:
n_jobs parallelism uses joblib/loky, which re-imports this module in each
worker - without the guard they'd re-run the search.
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import autotrainer


def main() -> None:
    X, y = make_classification(n_samples=5000, n_features=20)

    # Works on plain estimators and on nested ones (pipeline inside grid search)
    pipe = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier())])
    search = GridSearchCV(pipe, {"rf__n_estimators": [100, 200], "rf__max_depth": [None, 10]})
    search = autotrainer.prepare(search)  # n_jobs set everywhere it applies

    search.fit(X, y)
    print("best score:", search.best_score_)


if __name__ == "__main__":
    main()
