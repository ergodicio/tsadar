#!/bin/bash
#SBATCH -A m4490_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=4

unset JAX_PLATFORMS
export SLURM_CPU_BIND="cores"
export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="https://continuum.ergodic.io/experiments/"
# Coalesce the per-epoch async log_metrics calls (loops.py's _log_optimizer_step) into
# batched requests instead of one WAN round trip per epoch -- without this, MLflow's
# async queue submits each call as its own request and the backlog has to fully drain
# (blocking) when the run ends, showing up as a multi-minute hang after fitting finishes.
export MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS=5

# copy job stuff over
cd /global/homes/a/amilder/inverse-thomson-scattering
source .venv/bin/activate