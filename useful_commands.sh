# rsync all saved files
rsync --verbose --human-readable --progress --archive --compress --delete \
    ody:/n/regal/pillai_lab/mrischard/temperature_model/saved /Volumes/Samsung250GB/temperature_model/

# rsync just stan samples
rsync --verbose --human-readable --progress --archive --compress --delete \
    ody:/n/regal/pillai_lab/mrischard/temperature_model/saved/stan_fit/ /Volumes/Samsung250GB/temperature_model/saved/stan_fit

# render PDF
jupyter nbconvert --to latex --template nocode.tplx TemperatureImputations.ipynb
latexmk -bibtex -pdf TemperatureImputations