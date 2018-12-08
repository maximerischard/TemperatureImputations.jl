doc = """
    Impute hourly temperatures based on nearby hourly temperature records and on
    local Tn and Tx records. The posterior based on nearby hourly temperatures only
    has already been computed, with posterior means and covariances saved
    in saved/predictions_from_nearby.
    This script constrains those posteriors to be within the measured Tn&Tx.

    Usage:
        pipeline2.jl <ICAO> <model> <windownum> <data_dir> <save_dir> [--cheat]
"""
using DocOpt
arguments = docopt(doc)

save_dir = arguments["<save_dir>"]
save_dir = joinpath(save_dir)
println("directory for saved files: ", save_dir)
data_dir = arguments["<data_dir>"]
data_dir = joinpath(data_dir)
windownum = parse(Int, arguments["<windownum>"])
ICAO = arguments["<ICAO>"]
@show ICAO
GPmodel = arguments["<model>"]
@show GPmodel
cheat = arguments["--cheat"]
@show cheat

using CmdStan
using Dates
using Dates: Day, Hour, Date, DateTime
using JLD
using CSV
using TempModel
using LinearAlgebra
using PDMats
using DataFrames
using Printf
using Statistics

stan_days = Day(9)
stan_increment = Day(3)

isdList = dropmissing(TempModel.read_isdList(; data_dir=data_dir))
test_station = isdList[isdList[:ICAO].==ICAO, :]
@assert nrow(test_station) == 1
USAF = test_station[1, :USAF]
WBAN = test_station[1, :WBAN]
@show USAF
@show WBAN
hr_measure = Hour(17)

let
    hourly_test=TempModel.read_Stations(test_station; data_dir=data_dir)
    itest=1
    global TnTx = TempModel.test_data(hourly_test, itest, hr_measure)
end

struct FittingWindow
    start_date::DateTime
    end_date::DateTime
end

function predictions_fname(usaf::Int, wban::Int, icao::String, fw::FittingWindow)
     @sprintf("%d_%d_%s_%s_to_%s.jld", 
        usaf, wban, icao,
        Date(fw.start_date), Date(fw.end_date))
end

# copy-pasted from pipeline1.jl
nearby_windows = FittingWindow[]
mintime = DateTime(2015,1,1,0,0,0)
maxtime = DateTime(2016,1,1,0,0,0)
increm=(maxtime-mintime) / 15
dt_start = mintime
window=3*increm
while true
	global dt_start
    dt_end=dt_start+window
    fwindow = FittingWindow(dt_start,dt_end)
    push!(nearby_windows, fwindow)
    if dt_end >= maxtime
        break
    end
    dt_start+=increm
end

""" 
    Is window A inside of window B?
"""
function a_isinside_b(a::FittingWindow, b::FittingWindow)
    start_after = a.start_date >= b.start_date
    end_before = a.end_date <= b.end_date
    return start_after & end_before
end
function overlap(a::FittingWindow, b::FittingWindow)
	a_after_b = a.start_date >= b.end_date
	b_after_a = b.start_date >= a.end_date
	return !(a_after_b || b_after_a)
end
"""
    How much buffer time is there on either side of the window?
"""
function buffer(a::FittingWindow, b::FittingWindow)
    start_diff = abs(a.start_date - b.start_date)
    end_diff = abs(a.end_date - b.end_date)
    return min(start_diff, end_diff)
end
""" 
    Amongst a list of candidate windows `cand`, find the window that includes `wind`
    with the largest buffer on either sides.
"""
function find_best_window(wind::FittingWindow, cands::Vector{FittingWindow})
    incl_wdows = [fw for fw in cands if overlap(wind, fw)]
    buffers = [buffer(wind, fw) for fw in incl_wdows]
    imax = argmax(buffers)
    best_window = incl_wdows[imax]
    return best_window
end

janfirst = DateTime(2014,12,31,hr_measure.value,0,0)
stan_start = janfirst + (windownum-1)*stan_increment
stan_end = stan_start + stan_days
stan_window = FittingWindow(stan_start, stan_end)
println("STAN fitting window: ", stan_window)

best_window = find_best_window(stan_window, nearby_windows)
println("using nearby-predictions from: ", best_window)

nearby_path = joinpath(save_dir, "predictions_from_nearby", GPmodel,
                       ICAO, predictions_fname(USAF, WBAN, ICAO, best_window))
@show nearby_path
nearby_pred=load(nearby_path)["nearby_pred"];
"""
    In order to test whether the overestimated variance is causing
    the bias in yearly imputed mean temperatures, I'm going
    to rescale the predictive variances down so the
    standardized errors have variance 1.
    The function name is meant as a reminder that this is not
    an actual solution, just a diagnostic, as it requires
    access to the very truth that we are ultimately trying to impute.
"""
function variancecheat(nearby::TempModel.NearbyPrediction, truth::DataFrame)
    μ, Σ, nearbyts = nearby.μ, nearby.Σ, nearby.ts
    nobsv = length(nearbyts)
    centering = Matrix(1.0I, nobsv, nobsv) .- (1.0/nobsv)
    Σcentered = centering * Matrix(Σ) * centering
    Σmeanshift = Matrix(Σ) .- Σcentered
    
    # find the corresponding indices in `truth`
    ifirst = findfirst(isequal(nearbyts[1]), truth[:ts])
    ilast = ifirst + nobsv - 1
    @assert nearbyts[end] == truth[:ts][ilast]
    
    error = truth[ifirst:ilast,:temp] .- μ
    error .-= mean(error)
    stdpred = sqrt.(diag(Σcentered))
    sigma = error ./ stdpred
    @show mean(sigma), std(sigma) # should be 0,1
    scale = var(sigma)
    Σscaled = scale .* Σcentered
    
    # add some variance back in for an overall shift
    Σreshifted = Σscaled .+ mean(Σmeanshift) + 1e-12*I
    return TempModel.NearbyPrediction(nearbyts, μ, PDMats.PDMat(Symmetric(Σreshifted)))
end
if cheat
    @warn("THIS IS CHEATING!")
    truth = TempModel.read_Stations(test_station; data_dir=data_dir)
    global nearby_pred = variancecheat(nearby_pred, truth)
end

start_date = Date(stan_window.start_date)+Day(1) # date of first measurement
imputation_data, ts_window = TempModel.prep_data(nearby_pred, TnTx, start_date, hr_measure, stan_days)

function stan_dirname(usaf::Int, wban::Int, icao::String, fw::FittingWindow)
    return @sprintf("%s/%d_%d_%s_%s_to_%s/", 
                    icao, usaf, wban, icao, Date(fw.start_date), Date(fw.end_date))
end

stan_dir = joinpath(save_dir,"stan_fit", GPmodel, stan_dirname(USAF, WBAN, ICAO, stan_window))
if cheat
    stan_dir = joinpath(save_dir,"stan_fit", GPmodel, "cheat", stan_dirname(USAF, WBAN, ICAO, stan_window))
end
if !isdir(stan_dir)
    mkpath(stan_dir)
end
imputation_model = TempModel.get_imputation_model(; pdir=stan_dir)
for fname in readdir(joinpath(stan_dir, "tmp"))
    mv(joinpath(stan_dir, "tmp", fname), joinpath(stan_dir, fname); force=true)
end
rm(joinpath(stan_dir, "tmp"))

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

imputation_model.tmpdir = stan_dir;

@time sim1 = stan(
    imputation_model, 
    [imputation_data], 
    stan_dir,
    summary=false, 
    diagnostics=false
    )
println("=========")
