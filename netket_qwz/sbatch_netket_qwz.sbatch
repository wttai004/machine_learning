#!/bin/bash

#SBATCH -J nkqwz
#SBATCH --ntasks=6
#SBATCH --nodes=6
#SBATCH --time=48:00:00

module load gcc mkl

source /home1/wttai/miniconda3/etc/profile.d/conda.sh

conda activate netket

srun python3 ~/multirun.py netket_qwz.task

rm netket_qwz.sync


