#!/bin/bash

#SBATCH -J nkqwz
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

module load gcc mkl

source /home1/wttai/miniconda3/etc/profile.d/conda.sh

conda activate netket

srun python3 qwz_model.py
