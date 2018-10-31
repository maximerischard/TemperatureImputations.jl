#!/bin/bash
#SBATCH -J pipeline_kernelfit # name for job array
#SBATCH -p shared             # Partition
#SBATCH -t 0-24:00:00         # Running time of 24 hours
#SBATCH --mem 16000           # request 16GB memory
#SBATCH -n 2                  # 2 cores
#SBATCH -N 1                  # All cores on one machine
#SBATCH --mail-type=ALL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mrischard@g.harvard.edu
#SBATCH --array 32,47       # 27,28,29,35,36,38,39          # identifies job within batch

source ~/julia_modules.sh
test_ICAOs=(KWRB KMXF KJAN KESF KSJT KPHX KPOB KCAE KBNA KLRF KTIK KABQ KFAT 
            KDCA KBWI KDOV KWRI KLYH KEKN KFTK KCMH KIND KSPI KCOU KICT KHVE 
            KTPH KPVD KBDL KAOO KSYR KDSM KLBF KCPR KCON KMPV KMLT KMKG KVOK
            KHON KAXN KLWT KMUO KBNO KBIS KYKM KCEF KAFF)
ICAO=${test_ICAOs[$SLURM_ARRAY_TASK_ID-1]}
cd /n/regal/pillai_lab/mrischard/TempModel/batch/
echo "command line arguments"
echo "GPmodel" $1
echo "ICAO" $ICAO
julia pipeline0_kernelfit.jl $ICAO $1 /n/regal/pillai_lab/mrischard/TempModel /n/regal/pillai_lab/mrischard/saved --knearest=5

