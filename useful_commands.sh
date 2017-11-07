# rsync all saved files
rsync --verbose --human-readable --progress --archive --compress --delete \
    ody:/n/regal/pillai_lab/mrischard/temperature_model/saved /Volumes/Samsung250GB/temperature_model/

# rsync just stan samples
rsync --verbose --human-readable --progress --archive --compress --delete \
    ody:/n/regal/pillai_lab/mrischard/temperature_model/saved/stan_fit/ /Volumes/Samsung250GB/temperature_model/saved/stan_fit

# render PDF
source ~/bin/venv_nbconvert/bin/activate
jupyter nbconvert --to latex --template nocode.tplx TemperatureImputations.ipynb
latexmk -bibtex -pdf TemperatureImputations

# pipeline1.jl
srun -p interact --pty --mem 4000 -n 4 -N 1 -t 0-9:00 /bin/bash
module load gcc/7.1.0-fasrc01 julia/0.6.0-fasrc01
cd /n/regal/pillai_lab/mrischard/temperature_model/batch/
julia pipeline1.jl simpler