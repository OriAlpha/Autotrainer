#!/bin/bash
# Multi-node validation run. Everything - environment, doctor, per-rank output,
# the checks - lands in ONE file you can hand over whole:
#
#     sbatch examples/slurm/validate_multinode.sbatch
#     # then, when it finishes:
#     cat autotrainer-validate-<jobid>.out
#
# Edit the four #SBATCH lines below for your cluster (partition/account too, if
# yours needs them). The only hard rule: --ntasks-per-node MUST equal the GPUs
# per node, because autotrainer runs one task per GPU.

#SBATCH --job-name=autotrainer-validate
#SBATCH --nodes=2                    # >=2, or this validates nothing
#SBATCH --ntasks-per-node=2          # MUST equal GPUs per node
#SBATCH --gres=gpu:2                 # GPUs per node
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=autotrainer-validate-%j.out
# stderr is intentionally not redirected: it merges into the file above, so
# tracebacks stay interleaved with the output that led to them.

set -x  # echo every command, so the log shows what actually ran

# --- your environment -------------------------------------------------
# source /path/to/.venv/bin/activate
# module load cuda

echo "=== 1. JOB ==="
date -Is
scontrol show job "$SLURM_JOB_ID"
echo "nodes: $SLURM_JOB_NODELIST   ntasks: $SLURM_NTASKS   cpus/task: $SLURM_CPUS_PER_TASK"

echo "=== 2. SOFTWARE ==="
which python
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version())"
python -c "import autotrainer; print('autotrainer', autotrainer.__version__, autotrainer.__file__)"

echo "=== 3. HARDWARE, PER NODE ==="
srun --ntasks-per-node=1 --label bash -c 'hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv; echo "ib:"; ls /sys/class/infiniband 2>/dev/null || echo "  none"'

echo "=== 4. AUTOTRAINER DOCTOR (per node) ==="
srun --ntasks-per-node=1 --label autotrainer doctor

echo "=== 5. AUTOTRAINER INFO (every task) ==="
srun --label autotrainer info

echo "=== 6. THE VALIDATION RUN ==="
# --label prefixes every line with its task id - without it, interleaved
# multi-node output is unreadable and you cannot tell which rank hung.
srun --label autotrainer run examples/slurm/validate_multinode.py
echo "=== EXIT CODE: $? ==="

echo "=== 7. NCCL DEBUG RERUN (only useful if section 6 failed or hung) ==="
# Comment this out once section 6 passes; it is verbose. If the run above hung
# at a collective, this is the output that says which interface it tried.
# export NCCL_DEBUG=INFO
# srun --label autotrainer run examples/slurm/validate_multinode.py

date -Is
