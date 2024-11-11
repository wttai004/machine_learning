#!/bin/bash

#SBATCH -J dmrgqwz
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --time=48:00:00

module load gcc mkl

source /home1/wttai/miniconda3/etc/profile.d/conda.sh

conda activate /data1/conda/mobius

srun python3 ~/multirun.py dmrg_qwz.task --unique_output_dir

rm dmrg_qwz.sync


