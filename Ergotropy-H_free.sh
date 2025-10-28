#!/bin/bash
#SBATCH --job-name=ergotropy
#SBATCH --time=01:30:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # interne Parallelisierung -> 1 Task
#SBATCH --cpus-per-task=36         # 36 Kerne für deinen ProcessPool
#SBATCH --mem=468G                 # hoher RAM-Bedarf
#SBATCH --output=ergotropy_%j.out
#SBATCH --error=ergotropy_%j.err
#SBATCH --hint=nomultithread       # bevorzugt physische Kerne (wenn verfügbar)
# #SBATCH --exclusive              # optional: Node exklusiv reservieren

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CPUs pro Task: $SLURM_CPUS_PER_TASK"
echo "Node(s): $SLURM_NODELIST"

# --- Python/Conda Umgebung laden (bei dir anpassen) ---

# --- Numerik-Libs single-threaded, um Thread-Explosion zu vermeiden ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

# --- Anzahl interner Worker an Python weitergeben ---
export MAX_WORKERS=${SLURM_CPUS_PER_TASK}

echo "Starte Python-Job…"

# Sichere, portable Kern-Bindung (gute Cache-Lokalität)
# Falls dein System 'mem-bind' nicht unterstützt, NICHT verwenden.
SRUN_BIND="--cpu-bind=cores"

# Optional (bei sehr großen Matrizen >100 GB): NUMA-Interleave
# -> verteilt Allokationen über NUMA-Zonen (entweder ODER, nicht beides mischen)
# srun ${SRUN_BIND} numactl --interleave=all python Ergotropy-H_free.py && exit 0

srun ${SRUN_BIND} python Ergotropy-H_free.py

