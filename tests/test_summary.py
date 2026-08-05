import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import autotrainer  # noqa: E402
from autotrainer import summary as summary_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_global_summary():
    """Each test starts and ends with no process-global tracker, so one
    test's finish() can't suppress the next test's report."""
    summary_mod._ACTIVE_SUMMARY = None
    summary_mod._REPORT_EMITTED = False
    yield
    summary_mod._ACTIVE_SUMMARY = None
    summary_mod._REPORT_EMITTED = False


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


def test_autotrainer_finish(capsys):
    autotrainer.finish(checkpoint="dummy.pt")
    captured = capsys.readouterr().out
    assert "COMPREHENSIVE TRAINING SUMMARY" in captured or captured == ""


class TestAppliedSectionReportsOnlyWhatHappened:
    """The "Applied" section renders from the record the applying code wrote.

    The old version inferred it from torch globals and env vars, which cannot
    distinguish "autotrainer did this" from "the user did this" - so it
    claimed a DataLoader worker pipeline on loaders with num_workers=0, and a
    weight-decay param-group split that autotrainer has never performed.
    """

    def test_nothing_applied_prints_no_applied_section(self, capsys):
        autotrainer.SummaryTracker(total_samples=10, batch_size=2).report()
        captured = capsys.readouterr().out
        assert "Autotrainer Applied:" not in captured

    def test_dataloader_line_absent_when_loader_untouched(self, capsys):
        """No num_workers/pin_memory recorded -> no DataLoader claim."""
        summary = autotrainer.SummaryTracker(total_samples=10, batch_size=2)
        summary.record_applied(tf32=True)
        summary.report()
        captured = capsys.readouterr().out
        assert "TF32" in captured
        assert "DataLoader ->" not in captured

    def test_dataloader_line_present_when_recorded(self, capsys):
        summary = autotrainer.SummaryTracker(total_samples=10, batch_size=2)
        summary.record_applied(num_workers=4, pin_memory=True, persistent_workers=True)
        summary.report()
        captured = capsys.readouterr().out
        assert "num_workers=4" in captured
        assert "pin_memory" in captured

    def test_weight_decay_is_reported_not_claimed_as_optimization(self, capsys):
        """weight_decay shows up as the user's recipe, never as a param-group
        split autotrainer performed."""
        opt = torch.optim.AdamW([torch.randn(2)], lr=1e-3, weight_decay=0.01)
        summary = autotrainer.SummaryTracker(optimizer=opt)
        summary.report()
        captured = capsys.readouterr().out
        assert "weight_decay=0.01" in captured
        assert "Weight Decay Exclude" not in captured

    def test_no_dead_grad_clipping_claim(self, capsys):
        """`max_norm` is never an optimizer default - the old check was dead."""
        opt = torch.optim.SGD([torch.randn(2)], lr=0.1)
        assert "max_norm" not in opt.defaults
        summary = autotrainer.SummaryTracker(optimizer=opt)
        summary.report()
        assert "Grad Clipping" not in capsys.readouterr().out


class TestDetectedSectionDoesNotClaimCredit:
    """Env facts we observed but did not set are reported separately."""

    def test_preset_env_vars_land_under_detected(self, capsys, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

        autotrainer.SummaryTracker(total_samples=500, batch_size=32).report()
        captured = capsys.readouterr().out

        assert "Environment Detected:" in captured
        assert "NCCL_SOCKET_IFNAME=eth0" in captured
        assert "SLURM_CPUS_PER_TASK=8" in captured
        # Being inside a SLURM job says nothing about configure_scratch()
        # having been called, so no scratch claim.
        assert "Node scratch" not in captured

    def test_nccl_moves_to_applied_when_autotrainer_set_it(self, capsys, monkeypatch):
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ib0")
        summary = autotrainer.SummaryTracker()
        summary.record_applied(nccl_ifname="ib0")
        summary.report()
        captured = capsys.readouterr().out
        assert "NCCL_SOCKET_IFNAME -> set to ib0" in captured
        assert "preset; left alone" not in captured


class TestGlobalTrackerLifecycle:
    """finish() releases the global tracker so back-to-back runs each get a
    summary, and the atexit hook never prints an empty second box."""

    def test_second_run_in_one_process_gets_its_own_summary(self, capsys):
        autotrainer.log_epoch(train_loss=1.0)
        autotrainer.finish()
        first = capsys.readouterr().out
        assert "COMPREHENSIVE TRAINING SUMMARY" in first

        autotrainer.log_epoch(train_loss=0.5)
        autotrainer.finish()
        second = capsys.readouterr().out
        assert "COMPREHENSIVE TRAINING SUMMARY" in second
        assert "0.5000" in second

    def test_atexit_is_silent_after_explicit_finish(self, capsys):
        autotrainer.log_epoch(train_loss=1.0)
        autotrainer.finish()
        capsys.readouterr()
        summary_mod._on_exit()
        assert capsys.readouterr().out == ""

    def test_atexit_reports_when_finish_never_called(self, capsys):
        autotrainer.log_epoch(train_loss=1.0)
        summary_mod._on_exit()
        assert "COMPREHENSIVE TRAINING SUMMARY" in capsys.readouterr().out

    def test_atexit_does_not_destroy_process_group(self, monkeypatch):
        """Teardown at interpreter shutdown is unreliable; only an explicit
        finish(cleanup_dist=True) may do it."""
        destroyed = []
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        # report() reads the world size whenever the group looks initialized,
        # so the fake has to answer that too.
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: destroyed.append(1))
        autotrainer.log_epoch(train_loss=1.0)
        summary_mod._on_exit()
        assert destroyed == []

        autotrainer.log_epoch(train_loss=1.0)
        autotrainer.finish(cleanup_dist=True)
        assert destroyed == [1]
