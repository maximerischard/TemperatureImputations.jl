# rsync all saved files
rsync --verbose --human-readable --progress --archive --compress --update \
    ody:/n/regal/pillai_lab/mrischard/TempModel/saved /Volumes/Samsung250GB/TempModel/

# or
rsync --verbose --human-readable --progress --archive --compress \
    ody:/n/regal/pillai_lab/mrischard/TempModel/saved/stan_fit/ /Volumes/Samsung250GB/TempModel/saved/stan_fit

# rsync just stan samples for hour measurement
rsync --verbose --human-readable --progress --archive --compress \
    ody:/n/regal/pillai_lab/mrischard/TempModel/saved/hr_measure/ /Volumes/Samsung250GB/TempModel/saved/hr_measure

# note: --delete means “delete extraneous files from dest dirs”

# render PDF
source ~/bin/venv_nbconvert/bin/activate
jupyter nbconvert --to latex --template nocode.tplx TemperatureImputations.ipynb
latexmk -gg -bibtex -xelatex TemperatureImputations.tex

# pipeline0
# locally:
julia -e 'import Pkg; Pkg.activate("climate"; shared=true)' -L pipeline0_kernelfit.jl KWRB matern ../data ../notebooks --knearest=2
#SLURM:
sbatch --account=huybers_lab /n/regal/pillai_lab/mrischard/TempModel/batch/pipeline0_kernelfit.slurm matern

# pipeline1.jl
srun -p shared --pty --mem 8000 -n 4 -N 1 -t 0-9:00 /bin/bash

source ~/julia_modules.sh

cd /n/regal/pillai_lab/mrischard/TempModel/batch/
julia pipeline1.jl diurnal

# pipeline2
cd /n/regal/pillai_lab/mrischard/TempModel/batch/
julia pipeline2.jl /n/regal/pillai_lab/mrischard/TempModel/saved ${SLURM_ARRAY_TASK_ID} simpler
cd ~/logs && sbatch /n/regal/pillai_lab/mrischard/TempModel/batch/pipeline2.slurm matern KWRB

# Stan on Odyssey
# had to edit make/local to prevent stan from trying to use clang++ (which isn't available on odyssey)
# other than that, I'm using the downloaded binaries of stan 2.17.0 rather than the cmdstan module available on odyssey
CC=g++
CXX=g++

# pipeline_hr
source ~/julia_modules.sh
cd /n/regal/pillai_lab/mrischard/TempModel/batch/
julia pipeline_hr.jl /n/regal/pillai_lab/mrischard/TempModel/saved 42 simpler 17 5

cd ~/logs/
for hr in {0..23}
do
  sbatch /n/regal/pillai_lab/mrischard/TempModel/batch/pipeline_hr.slurm simpler $hr
done

# copy from home to regal
rsync --archive --verbose --human-readable --update ~/julia_lib/ /n/regal/pillai_lab/mrischard/julia_lib
rsync --archive --verbose --human-readable --update ~/cmdstan-2.17.0/ /n/regal/pillai_lab/mrischard/cmdstan-2.17.0
rsync --archive --verbose --human-readable --update ~/TempModel/ /n/regal/pillai_lab/mrischard/TempModel
# backup from regal to home
rsync --archive --verbose --human-readable --update /n/regal/pillai_lab/mrischard/julia_lib/ ~/julia_lib
rsync --archive --verbose --human-readable --update /n/regal/pillai_lab/mrischard/cmdstan-2.17.0/ ~/cmdstan-2.17.0
rsync --archive --verbose --human-readable --update /n/regal/pillai_lab/mrischard/TempModel/ ~/TempModel

# log into github
ssh -T git@github.com

# debugging Stan
# delete tmp files
rm -rf $mrischard/TempModel/tmp/imputation*
# delete existing Stan files, e.g.
rm -rf $mrischard/TempModel/saved/hr_measure/simpler/5/725480_2015-05-04_to_2015-05-13/imputation*
