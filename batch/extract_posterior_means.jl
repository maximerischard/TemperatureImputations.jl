doc = """
    * Extract the posterior mean and covariance from the imputations.

    Usage:
        extract_posterior_means.jl <ICAO> <model> <data_dir> <save_dir> [--hr_measure=<hr>] [--cheat]

    Options:
        -h --help     Show this screen.
        --hr_measure=<hr> Hour of daily measurement in UTC timezone [default: 17]
"""
using DocOpt
import TempModel
# using Printf: @printf, @sprintf
import JSON
using Dates
using DataFrames
using Statistics


arguments = docopt(doc)
ICAO = arguments["<ICAO>"]
@show ICAO
GPmodel = arguments["<model>"]
@show GPmodel
data_dir= joinpath(arguments["<data_dir>"])
@show data_dir
save_dir= joinpath(arguments["<save_dir>"])
@show save_dir
hr_measure_str = arguments["--hr_measure"]
if hr_measure_str === nothing
    hr_measure = Hour(17)
else
    hr_measure = Hour(parse(Int, hr_measure_str))
end
@show hr_measure

module Batch
    using ..TempModel
    using DataFrames
    using Dates
    using Distributions
    using PDMats
    using LinearAlgebra
    using Printf
    using Statistics
    src_dir = dirname(pathof(TempModel))
    include(joinpath(src_dir, "batch_diagnostics.jl"))
    include(joinpath(src_dir, "infermean.jl"))
end

epsg = 3857 # Web Mercator (m)
isdList = TempModel.read_isdList(; data_dir=data_dir, epsg=epsg)
isd_wData = TempModel.stations_with_data(isdList; data_dir=data_dir)

test_station = isd_wData[isd_wData[:ICAO].==ICAO, :]
@assert nrow(test_station) == 1
USAF = test_station[1, :USAF]
WBAN = test_station[1, :WBAN]

all_posteriors = Dict{Symbol}[]
for wnum in 1:119
    print(".$(wnum)")
    stan_fw = Batch.get_window(wnum)
    chains, _ts = Batch.get_chains_and_ts(stan_fw, GPmodel, USAF, WBAN, ICAO, save_dir)
    
    days, means_by_day = Batch.get_means_by_day(chains, _ts, hr_measure)

    nsamples, nchains, ndays = size(means_by_day)
    means_by_day_cat = reshape(means_by_day, (nsamples*nchains, ndays))
    post_cov   = cov(means_by_day_cat; dims=1)
    post_mean = vec(mean(means_by_day_cat; dims=1))
    push!(all_posteriors, Dict(:days=>days, :mean=>post_mean, :cov=>post_cov))
end

out_save_dir = joinpath(save_dir, "daily_mean", GPmodel)
mkpath(out_save_dir)
filepath = joinpath(out_save_dir, "daily_means_$(ICAO).json")
open(filepath, "w") do io
    indent = 4
    JSON.print(io, all_posteriors, indent)
end
