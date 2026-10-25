#!/bin/bash
# Submits extraction, then the pilot analysis once extraction succeeds.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
jid=$(sbatch --parsable slurm/extract.sbatch)
echo "extract: $jid"
pid=$(sbatch --parsable --dependency=afterok:"$jid" --export=ALL,STAGE=pilot slurm/probes.sbatch)
echo "probes (pilot): $pid, runs after $jid"
