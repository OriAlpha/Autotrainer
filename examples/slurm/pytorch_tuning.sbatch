#!/bin/bash
#SBATCH --job-name=autotrainer-tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00             # tuning = many short trainings; budget time
#SBATCH --output=%x-%j.out

# Activate your virtual environment if needed
# source .venv/bin/activate

srun autotrainer run examples/pytorch_tune.py
