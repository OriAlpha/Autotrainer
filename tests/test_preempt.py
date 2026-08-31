"""Tests for the preemption module.

A watched SIGUSR1/SIGTERM sets a flag - it must never raise, unwind, or lose
the original handler, since a training loop decides for itself when to stop.
"""

from __future__ import annotations

import signal

import pytest

from autotrainer import preempt


class TestPreemptModule:
    @pytest.fixture(autouse=True)
    def _clean(self):
        preempt.reset()
        yield
        preempt.reset()

    def test_flag_starts_clear(self):
        assert preempt.preempted() is False

    def test_handler_sets_the_flag_without_raising(self):
        preempt.watch()
        preempt._handler(signal.SIGTERM, None)  # what the OS would deliver
        assert preempt.preempted() is True

    def test_watch_installs_and_restore_puts_back(self):
        original = signal.getsignal(signal.SIGTERM)
        installed = preempt.watch()
        assert "SIGTERM" in installed
        assert signal.getsignal(signal.SIGTERM) is preempt._handler
        preempt.restore()
        assert signal.getsignal(signal.SIGTERM) is original

    def test_watch_is_idempotent(self):
        first = preempt.watch()
        second = preempt.watch()
        assert "SIGTERM" in first
        assert second == []  # nothing re-installed, original handler not lost

    def test_unknown_signal_names_are_skipped(self):
        # SIGUSR1 does not exist on Windows; asking for a missing signal must
        # not blow up a training run.
        assert preempt.watch(("SIGNOPE",)) == []

    def test_reset_clears_the_flag(self):
        preempt._handler(signal.SIGTERM, None)
        preempt.reset()
        assert preempt.preempted() is False
