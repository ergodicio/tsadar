#!/bin/bash
#SBATCH -A m4434_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="$PSCRATCH/mlflow"

# copy job stuff over

module load cudnn/8.9.3_cuda12.lua
module load cudatoolkit/12.0.lua
module load conda
conda activate ts

cd /global/u1/s/soham_b/tsadar_copy
srun python3 run.py
