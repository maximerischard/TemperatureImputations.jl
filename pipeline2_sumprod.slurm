#!/bin/bash
#SBATCH -J stan_pipelin2     # name for job array
#SBATCH -p general    #Partition
#SBATCH -t 0-12:00:00         #Running time of 1 day
#SBATCH --mem-per-cpu 2000   # request 4GB memory
#SBATCH -n 1                 # 2 cores
#SBATCH -N 1                 # All cores on one machine
#SBATCH --mail-type=ALL      #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu
#SBATCH --array 119,6,8,9,10,11,12,15,16,17,19,20,25,26,27,28,31,33,34,38,39,42,45,46,47,48,49,50,51,52,54,55,56,57,63,64,65,67,69,72,73,74,75,77,78,79,80,81,82,83,84,86,87,88,89,90,91,92,93,98,99,100,101,102,103,104,105,106,107,108,110,112,114,115,118          # identifies job within batch

source ~/julia_modules.sh
cd /n/regal/pillai_lab/mrischard/temperature_model/
julia pipeline2.jl /n/regal/pillai_lab/mrischard/temperature_model/saved ${SLURM_ARRAY_TASK_ID} sumprod

