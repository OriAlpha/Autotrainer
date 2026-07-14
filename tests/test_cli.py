"""Tests for the CLI entry point (cli.py).

Exercising `main()` in-process (rather than via subprocess) lets pytest-cov
attribute the cli.py lines to the coverage report. The existing subprocess
smoke tests in test_autotrainer.py stay as integration checks.
"""

from __future__ import annotations

import sys

import pytest

from autotrainer.cli import main


class TestCLIInfo:
    def test_info_prints_environment(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["autotrainer", "info"])
        # main() calls return (not sys.exit) for the info subcommand.
        main()
        out = capsys.readouterr().out
        assert "mode" in out
        assert "world size" in out


class TestCLIDoctor:
    def test_doctor_runs_and_exits_zero(self, capsys, monkeypatch):
        # Use an ephemeral port unlikely to be in use.
        monkeypatch.setenv("AUTOTRAINER_PORT", "49997")
        monkeypatch.setattr(sys, "argv", ["autotrainer", "doctor"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        assert "detected mode" in capsys.readouterr().out


class TestCLIRun:
    def test_run_dispatches_to_launch(self, monkeypatch):
        """`autotrainer run <script>` must hand off to launcher.launch()."""
        captured: dict = {}

        def fake_launch(script, script_args):
            captured["script"] = script
            captured["args"] = script_args
            return 0

        monkeypatch.setattr("autotrainer.cli.launch", fake_launch)
        monkeypatch.setattr(
            sys, "argv", ["autotrainer", "run", "train.py", "--epochs", "5", "--lr", "0.1"]
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        assert captured["script"] == "train.py"
        assert captured["args"] == ["--epochs", "5", "--lr", "0.1"]

    def test_run_requires_script(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["autotrainer", "run"])
        with pytest.raises(SystemExit):
            main()

    def test_no_subcommand_errors(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["autotrainer"])
        with pytest.raises(SystemExit):
            main()
