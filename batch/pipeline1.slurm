#!/bin/bash
#SBATCH -J pipeline_nearby   # name for job array
#SBATCH -p shared            # Partition
#SBATCH -t 0-12:00:00        # Running time of 12 hours
#SBATCH --mem 8000   # request 8GB memory
#SBATCH -n 4                 # 1 core
#SBATCH -N 1                 # All cores on one machine
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu

export JULIA_DEPOT_PATH="${HOME}/julia_depots/climate"
source ~/julia_modules.sh
cd /n/home04/mrischard/TempModel/batch/
echo "command line arguments"
echo "GPmodel" $1
julia pipeline1.jl \
    $1 \
    /n/home04/mrischard/TempModel/data \
    /n/scratchlfs/pillai_lab/mrischard/saved
