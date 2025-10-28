#!/bin/bash
#SBATCH --job-name=ergotropy
#SBATCH --time=5:30:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36         # statt 12x3 externe Tasks
#SBATCH --mem=468G
#SBATCH --output=optimale-param%j.out
#SBATCH --error=optimale-param%j.err

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MAX_WORKERS=$SLURM_CPUS_PER_TASK   # optional fürs Python

echo "CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

srun -n 1 python Optim_Param.py

