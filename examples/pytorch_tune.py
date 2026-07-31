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
    X, y = torch.randn(2000, 20), torch.randint(0, 5, (2000,))
    train = DataLoader(TensorDataset(X[:1600], y[:1600]), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(X[1600:], y[1600:]), batch_size=128)

    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))

    best_model, best_params, study = autotrainer.tune(
        model,
        train,
        val,
        trials=20,  # ASHA search over the recipe (lr, wd, opt, batch, schedule, ...)
        epochs_per_trial=3,
        save_path="best_model.pt",
    )
    print("Best params:", best_params)




if __name__ == "__main__":
    main()
