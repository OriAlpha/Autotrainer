"""Smart defaults: 1-line execution for end-to-end training.
Run: autotrainer run pytorch_auto.py

The ``if __name__ == "__main__":`` guard is required on Windows/macOS-spawn:
spawned DataLoader workers re-import this module, and without the guard they'd
re-run training instead of just loading data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer


def main() -> None:
    X, y = torch.randn(2048, 32), torch.randint(0, 10, (2048,))
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10))

    # EXACTLY 1 LINE: infers recipe, runs AMP loop, saves checkpoint, & prints summary!
    #
    # Supported kwargs for autotrainer.train():
    #   - epochs=10                          : Number of training epochs
    #   - save_path="model.pt"               : Auto-saves checkpoint (.pt, .joblib, .keras, .json)
    #   - lr=1e-3                            : Custom learning rate (or None to auto-tune via LR range test)
    #   - loss_fn=nn.CrossEntropyLoss()      : Custom loss function (or None to auto-infer from targets)
    #   - optimizer=optim.AdamW(model.parameters()) : Custom optimizer (or None for AdamW default)
    #   - patience=5                         : Early stopping patience (halts when loss stops improving)
    autotrainer.train(model, loader, epochs=3, save_path="model.pt")


if __name__ == "__main__":
    main()
