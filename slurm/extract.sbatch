#!/bin/bash
#SBATCH --job-name=sth-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# GPU: Qwen2.5-VL-7B hidden states for every image x prompt (~1500 forward passes, no generation).
# Pass LIMIT=20 for a smoke test: sbatch --export=ALL,LIMIT=20 slurm/extract.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export HF_HOME="$PWD/.hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/extract_activations.py --data data --out results/activations ${LIMIT:+--limit $LIMIT}
