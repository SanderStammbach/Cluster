#!/bin/bash
#SBATCH --job-name=ergotropy
#SBATCH --time=1:30:00                 # Laufzeitlimit (medium: bis 1-00:00:00 möglich)
#SBATCH --partition=short              # fat-node[001-008] sind in medium
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # EIN Prozess, intern parallelisiert
#SBATCH --cpus-per-task=36              # 36 CPUs für deinen ProcessPool
#SBATCH --mem=468G                      # >=468 GiB (fat-node hat ~480000 MB)
# # Optional: ganze Node exklusiv + voller RAM (falls gewünscht/erlaubt)
# #SBATCH --exclusive
# #SBATCH --mem=0
#SBATCH --output=battery_%j.out
#SBATCH --error=battery_%j.err

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node(s): $SLURM_NODELIST"
echo "CPUs/Task: $SLURM_CPUS_PER_TASK"
echo "Mem/Node (req): ${SLURM_MEM_PER_NODE:-unknown}"

# === Python/Conda Umgebung laden (bei euch anpassen) ===
# module load python/3.10
# conda activate qutip

# === Oversubscription vermeiden: BLAS/MKL Threads auf 1 ===
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optional: lokales Scratch nutzen (schnell & groß)
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export MPLCONFIGDIR="$TMPDIR"

# Optional: Anzahl Worker an dein Python übergeben
export MAX_WORKERS="${SLURM_CPUS_PER_TASK}"

# (Optional) Wechsel ins Verzeichnis mit deinem Skript
# cd /pfad/zu/deinem/projekt

echo "Starte Python-Job…"
# -u = unbuffered; --cpu-bind=cores bindet schön auf Kerne
srun --cpu-bind=cores python -u Ergotropy-Battery.py
