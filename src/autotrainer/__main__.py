"""Allow `python -m autotrainer <command>` as an alternative to the
console script - on HPC clusters the scripts directory is often not on
PATH (user installs, unactivated venvs), while `python -m` always works
with the interpreter that has the package.
"""

from .cli import main

if __name__ == "__main__":
    # main() exits the process itself (sys.exit) for run/doctor and returns
    # for info; just call it, matching cli.py's own __main__ block.
    main()
