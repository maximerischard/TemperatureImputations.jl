#!/bin/bash
#SBATCH -J stan_pipeline2    # name for job array
#SBATCH -p shared            # Partition
#SBATCH -t 0-12:00:00         # Running time of 1 hour
#SBATCH --mem 5000           # request 4GB memory
#SBATCH -n 2                 # 4 cores
#SBATCH -N 1                 # All cores on one machine
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu
#SBATCH --array 1-90         # identifies job within batch 

#test_ICAOs=(KABE KABQ KABR KATL KAUG KBDL KBHM KBIS KBNA KBWI KCAE KCEF KCMH
            #KCOS KCRW KDLH KDSM KEUG KFAT KFYV KGTF KICT KIND KINW KJAN KJAX
            #KLBF KLEX KLSE KMPV KMWL KOKC KPIH KPLN KPVD KRDU KROA KSEA KSGF
            #KSHV KSLC KSPI KSYR KTPH)
export JULIA_DEPOT_PATH="${HOME}/julia_depots/climate"
source ~/julia_modules.sh
echo "command line arguments"
GPmodel=$1
echo "GPmodel" $GPmodel
ICAO=$2
echo ICAO $ICAO
cd /n/home04/mrischard/TempModel/batch/
ksmoothmax=50
epsilon=0.01
julia pipeline2.jl \
    $ICAO \
    $GPmodel \
    ${SLURM_ARRAY_TASK_ID} \
    /n/home04/mrischard/TempModel/data \
    /n/scratchlfs/pillai_lab/mrischard/saved \
    --seed=${SLURM_ARRAY_TASK_ID} \
    --ksmoothmax=$ksmoothmax \
    --epsilon=$epsilon \
    --crossval
