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


class TestPythonDashM:
    def test_python_m_autotrainer_works(self):
        """`python -m autotrainer info` must work even when the console
        script is not on PATH (user installs / unactivated venvs on HPC)."""
        import subprocess

        r = subprocess.run(
            [sys.executable, "-m", "autotrainer", "info"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, r.stderr
        assert "mode" in r.stdout

    def test_main_module_in_process(self, monkeypatch):
        import runpy

        called = []
        monkeypatch.setattr("autotrainer.cli.main", lambda: called.append(True))
        runpy.run_module("autotrainer.__main__", run_name="__main__")
        assert called == [True]

        # Also exercise cli.py's own __name__ == '__main__' block
        import warnings

        monkeypatch.setattr(sys, "argv", ["autotrainer", "info"])
        monkeypatch.setattr(
            "autotrainer.cli.detect",
            lambda: type(
                "Env",
                (),
                {
                    "mode": "s",
                    "nnodes": 1,
                    "nproc_per_node": 1,
                    "world_size": 1,
                    "gpus": 0,
                    "master_addr": "localhost",
                    "master_port": 29500,
                    "notes": [],
                },
            )(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runpy.run_module("autotrainer.cli", run_name="__main__")


class TestCLICredits:
    def test_credits_prints_architecture_info(self, capsys, monkeypatch):
        for cmd in ("credits", "suhas"):
            monkeypatch.setattr(sys, "argv", ["autotrainer", cmd])
            main()
            out = capsys.readouterr().out
            assert "Autotrainer Core Architecture" in out
            assert "Suhas Goravale Siddaramu" in out


class TestCLIInfoNotes:
    def test_info_prints_notes(self, capsys, monkeypatch):
        from autotrainer.detect import Environment

        dummy_env = Environment(mode="single")
        dummy_env.notes.append("Custom diagnostic note for testing")
        monkeypatch.setattr("autotrainer.cli.detect", lambda: dummy_env)
        monkeypatch.setattr(sys, "argv", ["autotrainer", "info"])
        main()
        out = capsys.readouterr().out
        assert "Custom diagnostic note for testing" in out


class TestCLIUI:
    def test_ui_subcommand_dispatches_with_flags(self, monkeypatch):
        called_args = {}

        def fake_run_ui_server(logs_dirs, port, open_browser, host, token):
            called_args["logs_dirs"] = logs_dirs
            called_args["port"] = port
            called_args["open_browser"] = open_browser
            called_args["host"] = host
            called_args["token"] = token

        monkeypatch.setattr("autotrainer.ui.run_ui_server", fake_run_ui_server)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "autotrainer",
                "ui",
                "dir1",
                "dir2",
                "--port",
                "9876",
                "--host",
                "0.0.0.0",
                "--no-browser",
                "--no-token",
            ],
        )
        main()
        assert called_args["logs_dirs"] == ["dir1", "dir2"]
        assert called_args["port"] == 9876
        assert called_args["host"] == "0.0.0.0"
        assert called_args["open_browser"] is False
        assert called_args["token"] is None


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
