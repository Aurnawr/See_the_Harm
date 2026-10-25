#!/bin/bash
#SBATCH --job-name=sth-probes
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# CPU: low-level features, probes and report for one stage.
#   sbatch --export=ALL,STAGE=pilot slurm/probes.sbatch
#   sbatch --export=ALL,STAGE=final slurm/probes.sbatch   (only after the analysis plan is frozen and committed)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
STAGE="${STAGE:?set STAGE=pilot or STAGE=final}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[ -f results/lowlevel.npz ] || python scripts/lowlevel_features.py --data data --out results/lowlevel.npz
python scripts/probes.py --stage "$STAGE" --jobs "$SLURM_CPUS_PER_TASK"
python scripts/report.py --stage "$STAGE"
