#!/bin/bash
#SBATCH -J pipeline_nearby   # name for job array
#SBATCH -p shared            # Partition
#SBATCH -t 0-2:00:00        # Running time of 2 hours
#SBATCH --mem 12000          # request 16GB memory
#SBATCH -n 4                 # 4 cores
#SBATCH -N 1                 # All cores on one machine
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu
#SBATCH --array 1-2         # identifies job within batch

source ~/julia_modules.sh
test_ICAOs=(KWRB KMXF KJAN KESF KSJT KPHX KPOB KCAE KBNA KLRF KTIK KABQ KFAT 
            KDCA KBWI KDOV KWRI KLYH KEKN KFTK KCMH KIND KSPI KCOU KICT KHVE 
            KTPH KPVD KBDL KAOO KSYR KDSM KLBF KCPR KCON KMPV KMLT KMKG KVOK
            KHON KAXN KLWT KMUO KBNO KBIS KYKM KCEF KAFF)
ICAO=${test_ICAOs[$SLURM_ARRAY_TASK_ID-1]}
GPmodel=$1
cd /n/regal/pillai_lab/mrischard/TempModel/batch/
echo "command line arguments"
echo "GPmodel" $GPmodel
julia pipeline1.jl $ICAO $GPmodel /n/regal/pillai_lab/mrischard/TempModel/data /n/regal/pillai_lab/mrischard/saved --knearest=5
