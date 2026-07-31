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
    X, y = torch.randn(2048, 32), torch.randint(0, 10, (2048,))
    train = DataLoader(TensorDataset(X[:1600], y[:1600]), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(X[1600:], y[1600:]), batch_size=128)

    # Benchmark model (32 -> 128 -> 10) matched across all PyTorch examples
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10))

    # Phase 1: Optuna searches lr/weight_decay/optimizer/batch_size (short trials).
    # Phase 2: Winner retrained from original init with warmup+cosine, AMP, & early stopping.
    #
    # Supported options in autotrainer.fit():
    #   - trials=15                   : Optuna search trial count (ASHA pruned)
    #   - epochs=20                   : Max epochs for retraining winning recipe
    #   - patience=4                  : Early stopping patience epochs
    #   - metric="accuracy"           : Metric to select winner ("accuracy"/"f1"/"loss"/"auc")
    #   - save_path="fit_model.pt"    : Auto-saves rank-0 model checkpoint
    #   - test_loader=test_loader     : Held-out test set for un-biased evaluation
    #   - checkpoint="fit.ckpt"       : Resumable checkpoint file for SLURM preemption
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
