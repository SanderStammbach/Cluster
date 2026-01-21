#!/bin/bash
#SBATCH --job-name=TCL4
#SBATCH --time=48:00:00                 # Laufzeitlimit (medium: bis 1-00:00:00 möglich)
#SBATCH --partition=long              # fat-node[001-008] sind in medium
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # EIN Prozess, intern parallelisiert
#SBATCH --cpus-per-task=36              # 36 CPUs für deinen ProcessPool
#SBATCH --mem=120G                      # >=468 GiB (fat-node hat ~480000 MB)
#SBATCH --output=tcl4_%j.out
#SBATCH --error=tcl4_%j.err
# # Optional: ganze Node exklusiv + voller RAM (falls gewünscht/erlaubt)
# #SBATCH --exclusive
# #SBATCH --mem=0
set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

# --- Wichtig: verhindere Oversubscription durch BLAS/OpenMP Threads ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Python Output sofort flushen (hilft beim Debuggen) ---
export PYTHONUNBUFFERED=1

# --- falls dein Cluster module nutzt, ggf. aktivieren ---
# module purge
# module load python/3.10

# --- falls du ein venv benutzt ---
source ~/venvs2/qutip/bin/activate

# --- Pfad zu deinem Script ---
SCRIPT="TCL_2.py"

# Empfehlung: einmal Versionsinfo ausgeben
python - <<'PY'
import sys
print("Python:", sys.version)
try:
    import numpy, scipy
    print("numpy:", numpy.__version__, "scipy:", scipy.__version__)
except Exception as e:
    print("numpy/scipy import issue:", e)
try:
    import qutip
    print("qutip:", qutip.__version__)
except Exception as e:
    print("qutip import issue:", e)
PY

# --- Run (srun ist meist sauberer als direkt python) ---
srun --cpu-bind=cores python "$SCRIPT"

echo "Job finished at $(date)"

