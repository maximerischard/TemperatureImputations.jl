doc = """
    Impute hourly temperatures with the wrong measurement hour.
    The posterior based on nearby hourly temperatures only
    has already been computed, with posterior means and covariances saved
    in saved/predictions_from_nearby.
    This script constrains those posteriors to be within the measured Tn&Tx.

    Usage:
        pipeline2.jl <saved_dir> <windownum> <model> <hr_measure_true> <hr_measure_fals>
"""
using DocOpt

using Stan
using GaussianProcesses
using Proj4
using PDMats: PDMat
using Base.Dates: Day, Hour
using JLD

arguments = docopt(doc)
saved_dir = arguments["<saved_dir>"]
saved_dir = joinpath(saved_dir)
println("directory for saved files: ", saved_dir)
windownum = parse(Int, arguments["<windownum>"])
GPmodel = arguments["<model>"]
hr_measure_true = Hour(parse(Int, arguments["<hr_measure_true>"]))
hr_measure_fals = Hour(parse(Int, arguments["<hr_measure_fals>"]))

root_dir = ".."

stan_days = Day(9)
stan_increment = Day(3)

module TempModel
    root_dir = ".."
    using PDMats
    using PDMats: PDMat
    using GaussianProcesses: GP, Kernel, MeanZero, predict
    using Base.Dates: Day, Hour
    using Stan
    using DataFrames
    using DataFrames: by
    using GaussianProcesses
    using Proj4

    include(joinpath(root_dir, "src/utils.jl"))
    include(joinpath(root_dir, "src/preprocessing.jl"))
    include(joinpath(root_dir, "src/predict_from_nearby.jl"))
    include(joinpath(root_dir, "src/stan_impute.jl"))
end

type FittingWindow
    start_date::Date
    end_date::Date
end

function predictions_fname(usaf::Int, fw::FittingWindow)
    return @sprintf("%d_%s_to_%s.jld", 
                    usaf, fw.start_date, fw.end_date)
end

isdList=TempModel.read_isdList(;data_dir=root_dir)
isdSubset=isdList[[(usaf in (725450,725460,725480,725485)) for usaf in isdList[:USAF]],:]

hourly_cat=TempModel.read_Stations(isdSubset; data_dir=root_dir)
itest=3
test_usaf=isdSubset[itest,:USAF]

TnTx_true = TempModel.test_data(hourly_cat, itest, hr_measure_true)
TnTx_fals = TempModel.test_data(hourly_cat, itest, hr_measure_fals)
TnTx_fals[:Tn] = TnTx_true[:Tn] # corrupt TnTx
TnTx_fals[:Tx] = TnTx_true[:Tx]
;

# copy-pasted from pipeline1.jl
nearby_windows = FittingWindow[]
dt_start=DateTime(2015,1,1,0,0,0)
increm=(maximum(hourly_cat[:ts])-minimum(hourly_cat[:ts])) / 15
window=3*increm
while true
    dt_end=dt_start+window
    sdate = Date(dt_start)
    edate = Date(dt_end)
    fwindow = FittingWindow(sdate,edate)
    push!(nearby_windows, fwindow)
    if dt_end >= (maximum(hourly_cat[:ts]))
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
    incl_wdows = [fw for fw in cands if a_isinside_b(wind, fw)]
    buffers = [buffer(wind, fw) for fw in incl_wdows]
    imax = indmax(buffers)
    best_window = incl_wdows[imax]
    return best_window
end

janfirst = Date(2015,1,1)
stan_start = janfirst + (windownum-1)*stan_increment
stan_end = stan_start + stan_days
stan_window = FittingWindow(stan_start, stan_end)
println("STAN fitting window: ", stan_window)

best_window = find_best_window(stan_window, nearby_windows)
println("using nearby-predictions from: ", best_window)

nearby_pred=load(joinpath(saved_dir,
                        "predictions_from_nearby",
                        GPmodel,
                        predictions_fname(test_usaf, best_window),
                ))["nearby_pred"];

imputation_data, ts_window = TempModel.prep_data(nearby_pred, TnTx_fals, stan_window.start_date, hr_measure_fals, stan_days)

function stan_dirname(usaf::Int, fw::FittingWindow)
    return @sprintf("%d_%s_to_%s/", 
                    usaf, fw.start_date, fw.end_date)
end
stan_dir = joinpath(saved_dir,"hr_measure", GPmodel, string(Int(hr_measure_fals)), stan_dirname(test_usaf, stan_window))
if !isdir(stan_dir)
    mkpath(stan_dir)
end

imputation_model = TempModel.get_imputation_model(; pdir=stan_dir)
for fname in readdir(joinpath(stan_dir, "tmp"))
    mv(joinpath(stan_dir, "tmp", fname), joinpath(stan_dir, fname); remove_destination=true)
end
rm(joinpath(stan_dir, "tmp"))

writecsv(joinpath(stan_dir,"timestamps.csv"), reshape(ts_window, length(ts_window), 1))

for fname in ("imputation","imputation_build.log","imputation_run.log","imputation.hpp")
    tmpdir = joinpath(saved_dir, "..", "tmp")
    file_path = joinpath(tmpdir, fname)
    if isfile(file_path)
        cp(file_path, joinpath(stan_dir,fname), remove_destination=true)
    else
        println(file_path, "NOT FOUND")
    end
end
if isfile(joinpath(stan_dir,"imputation"))
    chmod(joinpath(stan_dir,"imputation"), 0o744)
end

imputation_model.tmpdir = stan_dir

@time sim1 = stan(
    imputation_model, 
    [imputation_data], 
    stan_dir,
    CmdStanDir=Stan.CMDSTAN_HOME, 
    summary=false, 
    diagnostics=false
    )
println("=========")
