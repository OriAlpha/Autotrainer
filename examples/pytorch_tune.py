"""Hyperparameter search: find the best training recipe for YOUR model.
Run: python pytorch_tune.py   (single process; GPU recommended)

The ``if __name__ == "__main__":`` guard is required on Windows/macOS-spawn:
each trial builds loaders with workers, and spawned workers re-import this
module - without the guard they'd re-launch the whole search.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer


def main() -> None:
    torch.manual_seed(0)
    X, y = torch.randn(2048, 32), torch.randint(0, 10, (2048,))
    train = DataLoader(TensorDataset(X[:1600], y[:1600]), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(X[1600:], y[1600:]), batch_size=128)

    # Benchmark model (32 -> 128 -> 10) matched across all PyTorch examples
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10))

    # EXACTLY 1 LINE: ASHA hyperparameter search over lr/decay/opt/batch/schedule
    #
    # Supported options in autotrainer.tune():
    #   - trials=20                   : Total Optuna search trials
    #   - epochs_per_trial=3          : Per-trial epoch budget
    #   - metric="loss"               : Metric to rank trials ("loss"/"accuracy"/"f1"/"auc")
    #   - save_path="best_model.pt"   : Auto-saves weights of the best trial
    #   - space=custom_dict           : Custom hyperparameter search space override
    #   - sanity=True                 : Warns about dataset leakage / un-normalized inputs
    best_model, best_params, study = autotrainer.tune(
        model,
        train,
        val,
        trials=20,
        epochs_per_trial=3,
        save_path="best_model.pt",
    )
    print("Best params:", best_params)


if __name__ == "__main__":
    main()
