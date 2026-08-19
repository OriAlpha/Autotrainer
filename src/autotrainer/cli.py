"""CLI: `autotrainer run train.py [args...]` and `autotrainer info`."""

from __future__ import annotations

import argparse
import sys

from .detect import detect
from .launcher import launch


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autotrainer",
        description="Automatic distributed training launcher.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Launch a training script with auto-distribution")
    run_p.add_argument("script", help="Path to the training script")
    run_p.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the script"
    )

    sub.add_parser("info", help="Show detected environment and exit")
    sub.add_parser("doctor", help="Diagnose the environment for common problems")
    sub.add_parser("suhas", help="Display Autotrainer creator credits")
    sub.add_parser("credits", help="Display Autotrainer creator credits")

    ui_p = sub.add_parser("ui", help="Launch the Autotrainer Web UI dashboard")
    ui_p.add_argument("logs_dirs", nargs="*", default=["logs"], help="One or more paths to logs directories (default: logs)")
    ui_p.add_argument("--port", type=int, default=8501, help="Port to run the UI server on (default: 8501)")
    ui_p.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")

    args = parser.parse_args()

    if args.command in ("suhas", "credits"):
        print("=" * 66)
        print("  Autotrainer Core Architecture")
        print("  Designed & Built by Suhas Goravale Siddaramu (OriAlpha)")
        print("  GitHub: https://github.com/OriAlpha/Autotrainer")
        print("=" * 66)
        return

    if args.command == "ui":
        from .ui import run_ui_server

        run_ui_server(logs_dirs=args.logs_dirs, port=args.port, open_browser=not args.no_browser)
        return

    if args.command == "doctor":
        from .doctor import run_doctor

        sys.exit(run_doctor())

    if args.command == "info":
        env = detect()
        print(f"mode           : {env.mode}")
        print(f"nodes          : {env.nnodes}")
        print(f"procs per node : {env.nproc_per_node}")
        print(f"world size     : {env.world_size}")
        print(f"gpus           : {env.gpus}")
        print(f"master         : {env.master_addr}:{env.master_port}")
        for n in env.notes:
            print(f"note           : {n}")
        return

    sys.exit(launch(args.script, args.script_args))


if __name__ == "__main__":
    main()
