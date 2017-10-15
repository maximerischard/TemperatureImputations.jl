#!/bin/bash
#SBATCH -J stan_pipeline2_SExSE     # name for job array
#SBATCH -p general    #Partition
#SBATCH -t 0-12:00:00         #Running time of 1 day
#SBATCH --mem-per-cpu 2000   # request 4GB memory
#SBATCH -n 1                 # 2 cores
#SBATCH -N 1                 # All cores on one machine
#SBATCH --mail-type=ALL      #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu
#SBATCH --array 1-119

source ~/julia_modules.sh
cd /n/regal/pillai_lab/mrischard/temperature_model/
julia pipeline2.jl /n/regal/pillai_lab/mrischard/temperature_model/saved ${SLURM_ARRAY_TASK_ID} SExSE

