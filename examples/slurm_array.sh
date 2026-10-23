#!/bin/bash
# Fit a large spectrum list as a SLURM array: 100 chunks, each on 32 cores.
#SBATCH --job-name=hiprofile
#SBATCH --array=0-99
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
set -euo pipefail
N_CHUNKS=100
hiprofile fit spectra.txt --config survey.yaml --out results \
    --jobs "${SLURM_CPUS_PER_TASK}" --chunk "${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}" --resume
# afterwards, once:  hiprofile report results
