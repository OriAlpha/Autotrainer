#!/bin/bash
#SBATCH --job-name=autotrainer-tf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1         # TF: ONE worker per node (drives all local GPUs)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# autotrainer generates TF_CONFIG from the SLURM node list automatically
srun autotrainer run tensorflow_scope.py
