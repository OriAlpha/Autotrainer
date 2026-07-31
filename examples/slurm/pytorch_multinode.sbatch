#!/bin/bash
#SBATCH --job-name=autotrainer-ddp
#SBATCH --nodes=2                   # scale freely - nothing else changes
#SBATCH --ntasks-per-node=4         # MUST equal GPUs per node (one task per GPU)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8           # dataloader workers per process
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Activate your virtual environment if needed
# source .venv/bin/activate

# Optional sanity check - fails fast if the environment is broken
srun autotrainer doctor

# Launch multi-node training
srun autotrainer run pytorch_auto.py
