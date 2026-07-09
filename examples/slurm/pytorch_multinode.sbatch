#!/bin/bash
#SBATCH --job-name=autotrainer-ddp
#SBATCH --nodes=2                   # scale freely - nothing else changes
#SBATCH --ntasks-per-node=4         # MUST equal GPUs per node (one task per GPU)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8           # dataloader workers per process
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Optional sanity check - fails fast if the environment is broken
autotrainer doctor

# srun spawns one process per task across all nodes; autotrainer reads the
# SLURM env vars to derive ranks and the rendezvous address automatically.
srun autotrainer run pytorch_auto.py
