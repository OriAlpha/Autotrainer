"""Example: autotrainer.train() with your OWN optimizer & loss function.
Detects hardware, sets up throughput optimizations (TF32, cudnn.benchmark, loader defaults, AMP),
while keeping your hyperparameter choices intact.

Run: autotrainer run pytorch_optimize.py
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # YOUR optimizer choice
    loss_fn = nn.CrossEntropyLoss()                           # YOUR loss choice

    # EXACTLY 1 LINE: Keeps your hyperparameters, turns on GPU hardware wins, runs AMP loop, & prints summary!
    autotrainer.train(model, loader, epochs=3, optimizer=optimizer, loss_fn=loss_fn, save_path="opt_model.pt")


if __name__ == "__main__":
    main()
