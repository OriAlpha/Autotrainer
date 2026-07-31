"""Tests for autotrainer.SummaryTracker."""

import torch

import autotrainer


def test_summary_tracker_basic():
    summary = autotrainer.SummaryTracker(total_samples=100, batch_size=10)
    summary.step(torch.tensor(0.5))
    summary.log_epoch(train_loss=0.5, val_loss=0.4, val_acc=90.0)
    summary.report()


def test_autotrainer_finish():
    autotrainer.finish(checkpoint="dummy.pt")

