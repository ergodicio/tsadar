#!/bin/bash
#SBATCH -A m4490_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="https://continuum.ergodic.io/experiments/"
export MLFLOW_EXPORT=True

# copy job stuff over
cd /global/homes/a/amilder/inverse-thomson-scattering
source .venv/bin/activate