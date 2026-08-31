"""Example: prepare() — YOUR training loop, on the hardware properly.

prepare() returns the same three objects you passed in, now distributed and
optimized (TF32, cudnn.benchmark, channels_last, loader workers, AMP). Your
lr, loss, schedule and optimizer are untouched. It is a no-op on CPU, so this
same script runs unchanged on a laptop and on an A100.

Reach for autotrainer.train() instead when you don't need your own loop —
see pytorch_auto.py.

Run: autotrainer run pytorch_optimize.py

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # YOUR optimizer choice
    loss_fn = nn.CrossEntropyLoss()                            # YOUR loss choice

    # ONE line: DDP wrap, device placement, sampler, and the throughput wins.
    # Pass optimize=False to opt out of the hardware flags and keep only the
    # distribution setup.
    model, loader, optimizer = autotrainer.prepare(model, loader, optimizer)

    # Created once, before the loop. A no-op on CPU and on bf16-capable GPUs.
    scaler = autotrainer.GradScaler()

    for epoch in range(3):
        autotrainer.set_epoch(loader, epoch)  # optional: prepare()'s loader
        model.train()                         # already reshuffles each epoch
        for xb, yb in loader:
            # train_step does the order-sensitive AMP dance for you: forward
            # under autocast, backward, step, update, zero_grad. Write it by
            # hand instead if you need something else in the middle.
            loss = autotrainer.train_step(model, loss_fn, xb, yb, optimizer, scaler=scaler)
        autotrainer.log_epoch(train_loss=float(loss))
        autotrainer.print0(f"epoch {epoch}: loss={float(loss):.4f}")

    # Saves from rank 0 only, then prints the training summary.
    autotrainer.save0(model.state_dict(), "opt_model.pt")
    autotrainer.finish(checkpoint="opt_model.pt", cleanup_dist=True)


if __name__ == "__main__":
    main()
