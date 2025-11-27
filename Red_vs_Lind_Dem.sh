#!/bin/bash
#SBATCH --job-name=ergotropy
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36         # statt 12x3 externe Tasks
#SBATCH --mem=120G
#SBATCH --output=Rotfeld%j.out
#SBATCH --error=Rotfeld%j.err

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MAX_WORKERS=$SLURM_CPUS_PER_TASK   # optional fürs Python

echo "CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

srun -n 1 python Red_vs_Lind_Dem.py

