"""One-call training: search the recipe, then fully train the winner.
Run: python pytorch_fit.py            (single device)
     autotrainer run pytorch_fit.py   (multi-GPU / SLURM: rank 0 tunes,
                                       every rank trains the winner via DDP)

The ``if __name__ == "__main__":`` guard is required on Windows/macOS-spawn:
fit() builds loaders with workers, and spawned workers re-import this module.
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

    # Phase 1: Optuna searches lr/weight_decay/optimizer/batch_size (short trials).
    # Phase 2: the winner is retrained from the original init - distributed when
    # launched with `autotrainer run` - with warmup+cosine, mixed precision, and
    # early stopping; the best epoch's weights come back.
    #
    # metric= is what both phases select on. It defaults to the val loss; name
    # the number you actually care about ("accuracy"/"f1"/"auc"/"r2", or your
    # own callable) and the search, the pruning, early stopping and best-epoch
    # selection all follow it. checkpoint= makes the run resumable AND makes it
    # stop cleanly on SLURM's preemption signal.
    model, params, study = autotrainer.fit(
        model,
        train,
        val,
        trials=15,
        epochs=20,
        patience=4,
        metric="accuracy",
        save_path="fit_model.pt",
    )
    print("Winning recipe:", params)




if __name__ == "__main__":
    main()
