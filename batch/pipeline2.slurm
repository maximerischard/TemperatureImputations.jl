#!/bin/bash
#SBATCH -J stan_pipeline2    # name for job array
#SBATCH -p shared            # Partition
#SBATCH -t 0-4:00:00         # Running time of 1 hour
#SBATCH --mem 3000           # request 4GB memory
#SBATCH -n 4                 # 4 cores
#SBATCH -N 1                 # All cores on one machine
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu
#SBATCH --array 1-119       # identifies job within batch 

#test_ICAOs=(KWRB KMXF KJAN KESF KSJT KPHX KPOB KCAE KBNA KLRF KTIK KABQ KFAT 
            #KDCA KBWI KDOV KWRI KLYH KEKN KFTK KCMH KIND KSPI KCOU KICT KHVE 
            #KTPH KPVD KBDL KAOO KSYR KDSM KLBF KCPR KCON KMPV KMLT KMKG KVOK
            #KHON KAXN KLWT KMUO KBNO KBIS KYKM KCEF KAFF)
source ~/julia_modules.sh
echo "command line arguments"
GPmodel=$1
echo "GPmodel" $GPmodel
ICAO=$2
echo ICAO $ICAO
cd /n/home04/mrischard/TempModel/batch/
julia pipeline2.jl $ICAO $GPmodel ${SLURM_ARRAY_TASK_ID} /n/home04/mrischard/TempModel/data /n/regal/pillai_lab/mrischard/saved 

