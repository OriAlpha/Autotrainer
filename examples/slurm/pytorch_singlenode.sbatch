#!/bin/bash
#SBATCH --job-name=autotrainer-1node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # = number of GPUs requested
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Activate your virtual environment if needed
# source .venv/bin/activate

# Optional sanity check - fails fast if the environment is broken
srun autotrainer doctor

# Launch single-node multi-GPU training
srun autotrainer run examples/pytorch_ddp.py
