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

srun autotrainer run examples/pytorch_ddp.py
