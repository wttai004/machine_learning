#!/bin/bash

#SBATCH -J nkqwz
#SBATCH --time=240:00:00

module load gcc mkl

source /home1/wttai/miniconda3/etc/profile.d/conda.sh

conda activate netket

srun python3 ~/multirun.py netket_qwz.task --unique_output_dir --create_database

rm netket_qwz.sync


