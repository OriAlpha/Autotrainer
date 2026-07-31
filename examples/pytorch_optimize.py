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
    #
    # Supported options in autotrainer.train():
    #   - epochs=3                    : Number of training epochs
    #   - optimizer=optimizer         : Custom optimizer instance
    #   - loss_fn=loss_fn             : Custom loss function instance
    #   - save_path="opt_model.pt"    : Auto-saves rank-0 checkpoint (.pt, .joblib, .keras, .json)
    #   - patience=5                  : Early stopping patience
    #   - lr=1e-3                     : Custom learning rate override
    autotrainer.train(model, loader, epochs=3, optimizer=optimizer, loss_fn=loss_fn, save_path="opt_model.pt")


if __name__ == "__main__":
    main()
