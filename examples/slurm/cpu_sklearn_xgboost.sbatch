#!/bin/bash
#SBATCH --job-name=autotrainer-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # autotrainer sets n_jobs to exactly this
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# No launcher needed - thread parallelism, not process parallelism
python sklearn_example.py
python xgboost_example.py
