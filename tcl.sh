#!/bin/bash
#SBATCH --job-name=tcl4
#SBATCH --output=tcl4_%j.out
#SBATCH --error=tcl4_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread

# Optional (je nach Cluster):
# #SBATCH --partition=compute
# #SBATCH --account=YOUR_ACCOUNT

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
SCRIPT="Tcl4-correlation-giantatom.py"

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

