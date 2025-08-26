#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT THESE ======
# NOTE: we no longer force-activate a conda env — the script will use the 'python'
# available in your current shell (so load modules / activate env before running
# the script if you want a specific python).
CODE_DIR="/scratch/uceerjp/gwdataset"   # use forward slashes, no trailing slash
# ========================

mkdir -p logs

# If you want the script to attempt to initialize conda (but not activate an env),
# uncomment the lines below. By default we do nothing and use current PATH/python.
# if command -v conda >/dev/null 2>&1; then
#   CONDA_BASE="$(conda info --base 2>/dev/null || true)"
#   if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
#     . "${CONDA_BASE}/etc/profile.d/conda.sh"
#   fi
# fi

# Determine worker count (use all logical cores by default; override with env GW_MAX_WORKERS)
if command -v nproc >/dev/null 2>&1; then
  export GW_MAX_WORKERS="$(nproc)"
else
  export GW_MAX_WORKERS="${GW_MAX_WORKERS:-0}"
fi

# Prevent math libs from oversubscribing each worker
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Headless-friendly + unbuffered logs
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Optional: increase ulimit if writing many images
ulimit -n 65535 || true

# cd safely (handle spaces)
cd "$CODE_DIR" || { echo "ERROR: CODE_DIR $CODE_DIR not found"; exit 2; }

LOG="logs/rsn_$(date +%Y%m%d_%H%M%S).log"
echo "Starting RSN job"
echo "  CODE_DIR:       $CODE_DIR"
echo "  LOG:            $LOG"
echo "  GW_MAX_WORKERS: ${GW_MAX_WORKERS}"
echo

# Launch using the 'python' currently in PATH.
# NOTE: script expects parallelMain.py (case sensitive) in CODE_DIR.
nohup python -u parallelmain.py > "$LOG" 2>&1 &

echo "Started PID $!  (tail -f \"$LOG\")"
