"""scikit-learn: auto n_jobs and 1-line execution.
Run: python sklearn.py

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

    pipe = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier())])
    search = GridSearchCV(pipe, {"rf__n_estimators": [100, 200], "rf__max_depth": [None, 10]})

    # ONE LINE: auto-configures n_jobs across CPU cores/SLURM tasks, fits, & serializes!
    #
    # Supported options for scikit-learn in autotrainer.train():
    #   - search                      : Any scikit-learn Estimator, Pipeline, or GridSearchCV
    #   - X, y                        : Feature matrix and target labels
    #   - save_path="search_model.joblib": Auto-saves model via joblib.dump()
    search = autotrainer.train(search, X, y, save_path="search_model.joblib")
    print("best score:", search.best_score_)


if __name__ == "__main__":
    main()
