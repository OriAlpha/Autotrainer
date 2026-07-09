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
    run_p.add_argument("script_args", nargs=argparse.REMAINDER,
                       help="Arguments forwarded to the script")

    sub.add_parser("info", help="Show detected environment and exit")
    sub.add_parser("doctor", help="Diagnose the environment for common problems")

    args = parser.parse_args()

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
