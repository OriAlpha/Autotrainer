"""Smart defaults: loss, optimizer, LR, and schedule inferred automatically.
Run: autotrainer run pytorch_auto.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer

X, y = torch.randn(2048, 32), torch.randint(0, 10, (2048,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10))

# One call: infers CrossEntropyLoss, AdamW, LR (range test), warmup+cosine
model, loader, opt, loss_fn, sched = autotrainer.auto(model, loader, epochs=3)
device = next(model.parameters()).device
scaler = autotrainer.GradScaler()  # handles mixed-precision scaling automatically

for epoch in range(3):
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        with autotrainer.autocast_context():          # mixed precision on GPU
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
    autotrainer.print0(f"epoch {epoch} done")          # prints once, not N times

autotrainer.save0(model.state_dict(), "model.pt")      # rank 0 saves
