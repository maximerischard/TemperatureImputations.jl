#!/bin/bash
#SBATCH -J stan_pipeline2    # name for job array
#SBATCH -p shared            # Partition
#SBATCH -t 0-1:00:00         # Running time of 1 hour
#SBATCH --mem 4000           # request 4GB memory
#SBATCH -n 4                 # 4 cores
#SBATCH -N 1                 # All cores on one machine
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu
#SBATCH --array 3-119        # identifies job within batch

source ~/julia_modules.sh
echo "command line arguments"
GPmodel=$1
echo "GPmodel" $GPmodel
ICAO=$2
echo ICAO $ICAO
cd /n/regal/pillai_lab/mrischard/TempModel/batch/
julia pipeline2.jl $ICAO $GPmodel ${SLURM_ARRAY_TASK_ID} /n/regal/pillai_lab/mrischard/TempModel/data /n/regal/pillai_lab/mrischard/saved 

