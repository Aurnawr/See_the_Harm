#!/bin/bash
# Login node, once. Network-bound downloads that compute nodes may not be able to do:
#   1. Qwen2.5-VL-7B-Instruct weights (~16 GB) into .hf_cache/
#   2. VLSU.csv and the RQ1 images into data/ (scripts/build_dataset.py, 4 download threads, no model)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME="$PWD/.hf_cache"
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
mkdir -p data
curl -fL -o data/VLSU.csv https://raw.githubusercontent.com/apple/ml-vlsu/main/data/VLSU.csv
python scripts/build_dataset.py --csv data/VLSU.csv --out data
