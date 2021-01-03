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
        --hr_measure_true=<hr>       [default: 17]
        --hr_measure_impt=<hr>      [default: 17]
"""
using DocOpt
arguments = docopt(doc)
import StanSample, MCMCChains
import Dates
using Dates: Day, Hour, Date, DateTime
using CSV
import TemperatureImputations
import PDMats, LinearAlgebra # needed for JLD.load to work, see https://github.com/JuliaIO/JLD.jl/issues/216
using DataFrames
using Printf
using Statistics


parse_if_not_empty(T, arg) = isnothing(arg) ? nothing : parse(T, arg)

save_dir = arguments["<save_dir>"]
save_dir = joinpath(save_dir)
@assert isdir(save_dir)
data_dir = arguments["<data_dir>"]
data_dir = joinpath(data_dir)
windownum = parse(Int, arguments["<windownum>"])
ICAO = arguments["<ICAO>"]
GPmodel = arguments["<model>"]
seed = parse(Int, arguments["--seed"])
ksmoothmax = parse_if_not_empty(Float64, arguments["--ksmoothmax"])
epsilon = parse_if_not_empty(Float64, arguments["--epsilon"])
crossval = arguments["--crossval"]::Bool
hr_measure_true = Hour(parse(Int, arguments["--hr_measure_true"]))
hr_measure_impt = Hour(parse(Int, arguments["--hr_measure_impt"]))

test_station_df = let
    epsg = 5072 # doesn't actually matter in this script
    isdList = dropmissing(TemperatureImputations.read_isdList(; epsg=epsg, data_dir=data_dir); disallowmissing=true)
    test_station_df = isdList[isdList.ICAO .== ICAO, :]
end

@assert nrow(test_station_df) == 1
test_station=test_station_df[1,:]
USAF = test_station.USAF
WBAN = test_station.WBAN

include("BatchTemperatureImputations.jl")

stan_days = Day(15)
stan_window = BatchTemperatureImputations.imputation_chunks(;stan_days=stan_days)[windownum]
stan_dir = BatchTemperatureImputations.stan_dirpath(;
    save_dir=save_dir,
    crossval=crossval,
    GPmodel=GPmodel,
    hr_measure=hr_measure_impt,
    usaf=USAF,
    wban=WBAN,
    icao=ICAO,
    fw=stan_window)
if !isdir(stan_dir)
    mkpath(stan_dir)
end
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
    TnTx_true = TemperatureImputations.test_data(hourly_test, itest, hr_measure_true)
    TnTx      = TemperatureImputations.test_data(hourly_test, itest, hr_measure_impt)
    TnTx[!,:Tn] = TnTx_true.Tn # corrupt TnTx
    TnTx[!,:Tx] = TnTx_true.Tx
    TnTx
end
;

nearby_windows = BatchTemperatureImputations.predict_from_nearby_chunks()

stan_window_with_times = BatchTemperatureImputations.add_time_to_window(stan_window, hr_measure_impt)
best_window = BatchTemperatureImputations.find_best_window(stan_window_with_times, nearby_windows)
println("using nearby-predictions from: ", best_window)

nearby_pred = BatchTemperatureImputations.load_predictions(;
    save_dir=save_dir,
    crossval=crossval,
    GPmodel=GPmodel,
    icao=ICAO,
    usaf=USAF,
    wban=WBAN,
    fw=best_window)

imputation_data, ts_window = TemperatureImputations.prep_data(
    nearby_pred,
    TnTx,
    Date(stan_window.start_date),
    hr_measure_impt,
    stan_days;
    ksmoothmax=ksmoothmax,
    epsilon=epsilon
)


CSV.write(joinpath(stan_dir,"timestamps.csv"), DataFrame(ts=ts_window), writeheader=false)

@time rc = StanSample.stan_sample(imputation_model, data=imputation_data)
@assert success(rc)

ENV["LINES"] = 2000
(StanSample.read_samples(imputation_model; output_format=:mcmcchains)
    |> x -> StanSample.summarystats(x)
    |> display
)
