"""Example: `prepare(optimize=True)` - detect hardware, set up throughput,
**leave the hyperparameters alone.**

This is the function the original autotrainer thesis describes. It turns on
the GPU wins users forget (TF32, cudnn.benchmark for CNNs, num_workers /
pin_memory / persistent_workers defaults on bare loaders, and AMP) without
touching lr / loss / schedule / optimizer. Run the same way as the other
examples:

    autotrainer run pytorch_optimize.py          # local, auto-detects GPUs
    srun autotrainer run pytorch_optimize.py     # inside an sbatch script
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import autotrainer

# Fake dataset (replace with your real one).
X = torch.randn(2048, 32)
y = torch.randint(0, 10, (2048,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # YOUR hyperparameter
loss_fn = nn.CrossEntropyLoss()  # YOUR hyperparameter

# ONE line detects hardware and sets TF32 / cudnn.benchmark / num_workers /
# pin_memory / persistent_workers / AMP. lr, loss, schedule, optimizer untouched.
# Add compile=True or fsdp=True as needed.
model, loader, optimizer = autotrainer.prepare(
    model,
    loader,
    optimizer,
    optimize=True,  # , compile=True, fsdp=True
)
device = next(model.parameters()).device
scaler = autotrainer.GradScaler()  # no-op when bf16 is available

for epoch in range(3):
    autotrainer.set_epoch(loader, epoch)  # reshuffles the DistributedSampler
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with autotrainer.autocast_context():  # bf16 if supported, else fp16
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        autotrainer.zero_grad(optimizer)  # set_to_none=True, saves memory
        total += loss.item()
    autotrainer.print0(f"epoch {epoch}: loss {total / len(loader):.4f}")
