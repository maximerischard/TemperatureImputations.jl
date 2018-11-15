#!/bin/bash
#SBATCH -J extract_posterior_means   # name for job array
#SBATCH -p general           # Partition
#SBATCH -t 0-0:20:00         # Running time of 20 minutes
#SBATCH --mem 2000           # request 2GB memory
#SBATCH -n 1                 # single core
#SBATCH -N 1                 # All cores on one machine
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu
#SBATCH --array 1,3,17,20,30,31,32,33,34,36,37,38,40,43

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
julia extract_posterior_means.jl $ICAO $GPmodel /n/home04/mrischard/TempModel/data /n/regal/pillai_lab/mrischard/saved --hr_measure=17
