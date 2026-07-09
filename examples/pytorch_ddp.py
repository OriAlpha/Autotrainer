"""Minimal example: a tiny model on random data, launched via `autotrainer run`."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer

# Fake dataset (replace with your real one)
X = torch.randn(2048, 32)
y = torch.randint(0, 10, (2048,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# The one autotrainer line: handles device placement, DDP, distributed sampler.
model, loader, optimizer = autotrainer.prepare(model, loader, optimizer)
device = next(model.parameters()).device

for epoch in range(3):
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)  # required for proper shuffling in DDP
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"epoch {epoch}: loss {total / len(loader):.4f}")
