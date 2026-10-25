#!/bin/bash
# Login node, once. Creates .venv/ in the repo and installs pinned dependencies (network only).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"'
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
