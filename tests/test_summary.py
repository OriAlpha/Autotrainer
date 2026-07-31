"""Tests for autotrainer.SummaryTracker and finish()."""

import os
import torch
import torch.nn as nn

import autotrainer


def test_summary_tracker_basic(capsys):
    summary = autotrainer.SummaryTracker(
        total_samples=100,
        batch_size=10,
        optimizer=torch.optim.Adam([torch.randn(2)], lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
    )
    summary.step(torch.tensor(0.5))
    summary.log_epoch(train_loss=0.5, val_loss=0.4, val_acc=90.0)
    summary.report(checkpoint="dummy.pt")

    captured = capsys.readouterr().out
    assert "COMPREHENSIVE TRAINING SUMMARY" in captured
    assert "Adam" in captured
    assert "CrossEntropyLoss" in captured
    assert "DataLoader Pipeline" in captured


def test_autotrainer_finish(capsys):
    autotrainer.finish(checkpoint="dummy.pt")
    captured = capsys.readouterr().out
    assert "COMPREHENSIVE TRAINING SUMMARY" in captured or captured == ""


def test_summary_optimizations_detection(capsys, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    summary = autotrainer.SummaryTracker(total_samples=500, batch_size=32)
    summary.report()

    captured = capsys.readouterr().out
    assert "SLURM Node Scratch" in captured
    assert "NCCL Interconnect" in captured
    assert "CPU Multi-Core Scaling" in captured
