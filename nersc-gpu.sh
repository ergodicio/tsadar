#!/bin/bash
#SBATCH -A m4490_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="https://continuum.ergodic.io/experiments/"
export MLFLOW_EXPORT=True

# copy job stuff over
module load python
module load cudnn/8.9.3_cuda12.lua
module load cudatoolkit/12.2.lua


conda activate tsadar-gpu
cd /global/homes/a/amilder/inverse-thomson-scattering