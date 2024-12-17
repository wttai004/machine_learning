#!/bin/bash

#SBATCH -J dmrgqwz
#SBATCH --ntasks=3
#SBTACH --cpus-per-task=128
#SBATCH --nodes=3
#SBATCH --time=96:00:00
#SBATCH --partition=highmem

module load gcc mkl

source /home1/wttai/miniconda3/etc/profile.d/conda.sh

conda activate /data1/conda/mobius

srun python3 ~/multirun.py dmrg_qwz.task --unique_output_dir

rm dmrg_qwz.sync


