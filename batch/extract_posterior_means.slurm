#!/bin/bash
#SBATCH -J extract_posterior_means   # name for job array
#SBATCH -p general           # Partition
#SBATCH -t 0-1:00:00         # Running time of 20 minutes
#SBATCH --mem 1000           # request 2GB memory
#SBATCH -n 1                 # single core
#SBATCH -N 1                 # All cores on one machine
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu
#SBATCH --array 1-44

export JULIA_DEPOT_PATH="${HOME}/julia_depots/climate"
source ~/julia_modules.sh
test_ICAOs=(KABE KABQ KABR KATL KAUG KBDL KBHM KBIS KBNA KBWI KCAE KCEF KCMH
            KCOS KCRW KDLH KDSM KEUG KFAT KFYV KGTF KICT KIND KINW KJAN KJAX
            KLBF KLEX KLSE KMPV KMWL KOKC KPIH KPLN KPVD KRDU KROA KSEA KSGF
            KSHV KSLC KSPI KSYR KTPH)
ICAO=${test_ICAOs[$SLURM_ARRAY_TASK_ID-1]}
GPmodel=$1
cd /n/home04/mrischard/TempModel/batch/
echo "command line arguments"
echo "GPmodel" $GPmodel
echo "ICAO" $ICAO
julia extract_posterior_means.jl \
    $ICAO \
    $GPmodel \
    /n/home04/mrischard/TempModel/data \
    /n/scratchlfs/pillai_lab/mrischard/saved \
    --hr_measure=17 \
    --crossval
