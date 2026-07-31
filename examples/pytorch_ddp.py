"""Minimal DDP training script with custom model, optimizer, & loss function.
Run: autotrainer run pytorch_ddp.py

The ``if __name__ == "__main__":`` guard is required on Windows/macOS-spawn:
DataLoader workers re-import this module, and without the guard they'd
re-run training instead of just loading data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer


def main() -> None:
    X = torch.randn(2048, 32)
    y = torch.randint(0, 10, (2048,))
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # EXACTLY 1 LINE: Handles DDP, device placement, AMP loop, checkpointing, & summary!
    autotrainer.train(model, loader, epochs=3, optimizer=optimizer, loss_fn=loss_fn, save_path="ddp_model.pt")


if __name__ == "__main__":
    main()
