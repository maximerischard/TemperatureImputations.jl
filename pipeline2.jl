doc = """
    Impute hourly temperatures based on nearby hourly temperature records and on
    local Tn and Tx records. The posterior based on nearby hourly temperatures only
    has already been computed, with posterior means and covariances saved
    in saved/predictions_from_nearby.
    This script constrains those posteriors to be within the measured Tn&Tx.

    Usage:
        pipeline2.jl <saved_dir> <windownum>
"""
using DocOpt
arguments = docopt(doc)

saved_dir = arguments["<saved_dir>"]
saved_dir = Pkg.dir(saved_dir)
if !endswith(saved_dir, "/")
    saved_dir = join((saved_dir,"/"))
end
println("directory for saved files: ", saved_dir)
windownum = parse(Int, arguments["<windownum>"])

using Stan
using DataFrames
using GaussianProcesses
using Proj4
using PDMats: PDMat
using DataFrames: head
using Base.Dates: Day, Hour
using JLD
using GaussianProcesses: SumKernel

stan_days = Day(9)
stan_increment = Day(3)

include("src/utils.jl")
include("src/preprocessing.jl")

isdList=read_isdList()
isdSubset=isdList[[(usaf in (725450,725460,725480,725485)) for usaf in isdList[:USAF].values],:]
isdSubset

hourly_cat=read_Stations(isdSubset)
itest=3
test_usaf=get(isdSubset[itest,:USAF])
hr_measure = Hour(17)

TnTx = test_data(hourly_cat, itest, hr_measure)

module TempModel
    using PDMats: PDMat
    using DataFrames
    using Mamba
    using GaussianProcesses: GP, Kernel, MeanZero, predict
    using Base.Dates: Day, Hour
    using Stan
    using DataFrames: DataFrame, by

    include("src/utils.jl")
    include("src/predict_from_nearby.jl")
    include("src/stan_impute.jl")
end

type FittingWindow
    start_date::Date
    end_date::Date
end

function predictions_fname(usaf::Int, fw::FittingWindow)
    return @sprintf("predictions_from_nearby/%d_%s_to_%s.jld", 
                    usaf, fw.start_date, fw.end_date)
end

# copy-pasted from pipeline1.jl
nearby_windows = FittingWindow[]
dt_start=DateTime(2015,1,1,0,0,0)
increm=get(maximum(hourly_cat[:ts])-minimum(hourly_cat[:ts])) / 15
window=3*increm
while true
    dt_end=dt_start+window
    sdate = Date(dt_start)
    edate = Date(dt_end)
    fwindow = FittingWindow(sdate,edate)
    push!(nearby_windows, fwindow)
    if dt_end >= get(maximum(hourly_cat[:ts]))
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

nearby_pred=load(join((saved_dir,predictions_fname(test_usaf, best_window))))["nearby_pred"];

imputation_data=TempModel.prep_data(nearby_pred, TnTx, stan_window.start_date, hr_measure, stan_days)

imputation_model = TempModel.get_imputation_model();

function stan_dirname(usaf::Int, fw::FittingWindow)
    return @sprintf("stan_fit/%d_%s_to_%s/", 
                    usaf, fw.start_date, fw.end_date)
end

stan_dir = join((saved_dir,stan_dirname(test_usaf, stan_window)))
dir_exists = isdir(stan_dir)
if !dir_exists
    mkdir(stan_dir)
end

for fname in ("imputation","imputation_build.log","imputation_run.log","imputation.hpp","imputation.stan")
    cp("tmp/imputation", join((stan_dir,fname)), remove_destination=true)
end
chmod(join((stan_dir,"imputation")), 0o744)

imputation_model.tmpdir = stan_dir;

@time sim1 = stan(
    imputation_model, 
    [imputation_data], 
    stan_dir,
    CmdStanDir=Stan.CMDSTAN_HOME, 
    summary=false, 
    diagnostics=false
    )
println("=========")
