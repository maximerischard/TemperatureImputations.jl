doc = """
    Impute hourly temperatures based on nearby hourly temperature records and on
    local Tn and Tx records. The posterior based on nearby hourly temperatures only
    has already been computed, with posterior means and covariances saved
    in saved/predictions_from_nearby.
    This script constrains those posteriors to be within the measured Tn&Tx.

    Usage:
        pipeline2.jl (outputdir|compilestan|impute) <ICAO> <model> <windownum> <data_dir> <save_dir> [options] [--crossval]

    Options:
        --seed=<seed>  Random seed for Stan.  [default: -1]
        --ksmoothmax=<ksmoothmax>
        --epsilon=<epsilon>
"""
using DocOpt
arguments = docopt(doc)
using StanSample
using Dates
using Dates: Day, Hour, Date, DateTime
using JLD
using CSV
import TemperatureImputations
using LinearAlgebra
using DataFrames
using Printf
using Statistics

function parse_if_not_empty(T, arg)
    if arg != ""
        return parse(T, arg)
    else
        return nothing
    end
end

save_dir = arguments["<save_dir>"]
save_dir = joinpath(save_dir)
@assert isdir(save_dir)
data_dir = arguments["<data_dir>"]
data_dir = joinpath(data_dir)
windownum = parse(Int, arguments["<windownum>"])
ICAO = arguments["<ICAO>"]
GPmodel = arguments["<model>"]
seed = parse_if_not_empty(Int, arguments["--seed"])
ksmoothmax = parse_if_not_empty(Float64, arguments["--ksmoothmax"])
epsilon = p_if_not_emptyarse(Float64, arguments["--epsilon"])
crossval = arguments["--crossval"]::Bool

test_station_df = let
    epsg = 5072 # doesn't actually matter in this script
    isdList = dropmissing(TemperatureImputations.read_isdList(; epsg=epsg, data_dir=data_dir); disallowmissing=true)
    test_station_df = isdList[isdList.ICAO .== ICAO, :]
end

@assert nrow(test_station_df) == 1
test_station=test_station_df[1,:]
USAF = test_station.USAF
WBAN = test_station.WBAN

struct FittingWindow
    start_date::DateTime
    end_date::DateTime
end
function stan_dirname(usaf::Int, wban::Int, icao::String, fw::FittingWindow)
    return @sprintf("%s/%d_%d_%s_%s_to_%s/", 
                    icao, usaf, wban, icao, Date(fw.start_date)+Day(1), Date(fw.end_date))
end

stan_days = Day(24)
hr_measure = Hour(17)
stan_increment = Day(14) # (24-14)/2 = 5 days of buffer
janfirst = DateTime(2014,12,31,hr_measure.value,0,0)
mintime = DateTime(2015,1,1,0,0,0)
maxtime = DateTime(2016,1,1,0,0,0)
stan_start = janfirst + (windownum-1)*stan_increment
stan_end = stan_start + stan_days
stan_window = FittingWindow(max(stan_start,mintime), min(stan_end,maxtime))
stan_dir = abspath(joinpath(
        save_dir, "stan_fit", 
        crossval ? "crossval" : "mll",
        GPmodel,
        stan_dirname(USAF, WBAN, ICAO, stan_window)
    ))
if arguments["outputdir"]
    println(stan_dir)
    exit()
end
if !isdir(stan_dir)
    mkpath(stan_dir)
end
imputation_model = TemperatureImputations.get_imputation_model(; pdir=stan_dir, seed=seed)
if arguments["compilestan"]
    exit()
end

TnTx = let
    hourly_test=TemperatureImputations.read_Stations(test_station_df; data_dir=data_dir)
    itest=1
    TemperatureImputations.test_data(hourly_test, itest, hr_measure)
end


function predictions_fname(usaf::Int, wban::Int, icao::String, fw::FittingWindow)
     @sprintf("%d_%d_%s_%s_to_%s.jld", 
        usaf, wban, icao,
        Date(fw.start_date), Date(fw.end_date))
end

# copy-pasted from pipeline1.jl
nearby_windows = FittingWindow[]
increm=(maxtime-mintime) / 15
dt_start = mintime
window=3*increm
while dt_start < maxtime
	global dt_start
    dt_end=dt_start+window
    fwindow = FittingWindow(dt_start,dt_end)
    push!(nearby_windows, fwindow)
    dt_start+=increm
    if dt_end >= maxtime
        break
    end
end

function overlap(a::FittingWindow, b::FittingWindow)
	# conditions that imply the windows don't overlap at all:
	a_after_b = a.start_date >= b.end_date
	b_after_a = b.start_date >= a.end_date
	return !(a_after_b || b_after_a)
end
"""
    How much buffer time is there on either side of the window?
"""
function buffer(a::FittingWindow, b::FittingWindow)
    start_diff = a.start_date - b.start_date
    end_diff = b.end_date - a.end_date
    return min(start_diff, end_diff) # worst of the two
end
""" 
    Amongst a list of candidate windows `cand`, find the window that includes `wind`
    with the largest buffer on either sides.
"""
function find_best_window(wind::FittingWindow, cands::Vector{FittingWindow})
    incl_wdows = [fw for fw in cands if overlap(wind, fw)]
    buffers = [buffer(wind, fw) for fw in incl_wdows]
    imax = argmax(buffers) # maximum of minimum
    best_window = incl_wdows[imax]
    return best_window
end


best_window = find_best_window(stan_window, nearby_windows)
println("using nearby-predictions from: ", best_window)

nearby_dir = joinpath(save_dir, "predictions_from_nearby", crossval ? "crossval" : "mll", GPmodel, ICAO)
nearby_path =  joinpath(nearby_dir, predictions_fname(USAF, WBAN, ICAO, best_window))
@show nearby_path
nearby_pred=load(nearby_path)["nearby_pred"];

start_date = Date(stan_window.start_date)+Day(1) # date of first measurement
imputation_data, ts_window = TemperatureImputations.prep_data(nearby_pred, TnTx, start_date, hr_measure, stan_days; ksmoothmax=ksmoothmax, epsilon=epsilon)


# for fname in readdir(joinpath(stan_dir, "tmp"))
    # mv(joinpath(stan_dir, "tmp", fname), joinpath(stan_dir, fname); force=true)
# end
# rm(joinpath(stan_dir, "tmp"))

CSV.write(joinpath(stan_dir,"timestamps.csv"), DataFrame(ts=ts_window), writeheader=false)

for fname in ("imputation","imputation_build.log","imputation_run.log","imputation.hpp")
    tmpdir = joinpath(save_dir, "tmp")
    file_path = joinpath(tmpdir, fname)
    if isfile(file_path)
        cp(file_path, joinpath(stan_dir,fname), force=true)
    else
        println(file_path, "NOT FOUND")
    end
end
if isfile(joinpath(stan_dir,"imputation"))
    chmod(joinpath(stan_dir,"imputation"), 0o744)
end

@time rc = StanSample.stan_sample(imputation_model, data=imputation_data)
@assert success(rc)

ENV["LINES"] = 2000
(StanSample.read_samples(imputation_model; output_format=:mcmcchains)
    |> x -> StanSample.summarystats(x)
    |> display
)
