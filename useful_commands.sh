# rsync all saved files
rsync --verbose --human-readable --progress --archive --compress --delete \
    ody:/n/regal/pillai_lab/mrischard/temperature_model/saved /Volumes/Samsung250GB/temperature_model/

# rsync just stan samples
rsync --verbose --human-readable --progress --archive --compress --delete \
    ody:/n/regal/pillai_lab/mrischard/temperature_model/saved/stan_fit/ /Volumes/Samsung250GB/temperature_model/saved/stan_fit
# or
rsync --verbose --human-readable --progress --archive --compress \
    ody:/n/regal/pillai_lab/mrischard/temperature_model/saved/stan_fit/ /Volumes/Samsung250GB/temperature_model/saved/stan_fit

# rsync just stan samples for hour measurement
rsync --verbose --human-readable --progress --archive --compress \
    ody:/n/regal/pillai_lab/mrischard/temperature_model/saved/hr_measure/ /Volumes/Samsung250GB/temperature_model/saved/hr_measure

# note: --delete means “delete extraneous files from dest dirs”

# render PDF
source ~/bin/venv_nbconvert/bin/activate
jupyter nbconvert --to latex --template nocode.tplx TemperatureImputations.ipynb
latexmk -bibtex -pdf TemperatureImputations

# pipeline1.jl
srun -p test --pty --mem 4000 -n 4 -N 1 -t 0-9:00 /bin/bash

module load gcc/7.1.0-fasrc01 
module load julia/0.6.0-fasrc01
module load OpenBLAS/0.2.18-fasrc01
module load git/2.1.0-fasrc01
# module load cmdstan/2.12.0-fasrc01

cd /n/regal/pillai_lab/mrischard/temperature_model/batch/
julia pipeline1.jl simpler

# pipeline2
cd /n/regal/pillai_lab/mrischard/temperature_model/batch/
julia pipeline2.jl /n/regal/pillai_lab/mrischard/temperature_model/saved ${SLURM_ARRAY_TASK_ID} simpler
cd ~/logs/
sbatch /n/regal/pillai_lab/mrischard/temperature_model/batch/pipeline2_simpler.slurm

# Stan on Odyssey
# had to edit make/local to prevent stan from trying to use clang++ (which isn't available on odyssey)
# other than that, I'm using the downloaded binaries of stan 2.17.0 rather than the cmdstan module available on odyssey
CC=g++
CXX=g++

# backup
rsync -av /n/regal/pillai_lab/mrischard/julia_lib/ ~/julia_lib


# LaTeX
source ~/bin/venv_nbconvert/bin/activate
jupyter nbconvert --to latex --template nocode.tplx TemperatureImputations.ipynb
latexmk -bibtex -pdf TemperatureImputations


# pipeline_hr
module load gcc/7.1.0-fasrc01 
module load julia/0.6.0-fasrc01
module load OpenBLAS/0.2.18-fasrc01

cd /n/regal/pillai_lab/mrischard/temperature_model/batch/
julia pipeline_hr.jl /n/regal/pillai_lab/mrischard/temperature_model/saved 42 simpler 17 5

cd ~/logs/
for hr in {0..23}
do
  sbatch /n/regal/pillai_lab/mrischard/temperature_model/batch/pipeline_hr.slurm simpler $hr
done