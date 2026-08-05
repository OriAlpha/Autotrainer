#!/bin/bash
#SBATCH --job-name=autotrainer-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # autotrainer sets n_jobs to exactly this
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Activate your virtual environment if needed
# source .venv/bin/activate

# Optional sanity check - fails fast if the environment is broken
autotrainer doctor

# Launch CPU training (thread parallelism, no launcher needed)
python examples/sklearn.py
python examples/xgboost.py
