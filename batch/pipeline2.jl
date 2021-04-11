doc = """
    Impute hourly temperatures based on nearby hourly temperature records and on
    local Tn and Tx records. The posterior based on nearby hourly temperatures only
    has already been computed, with posterior means and covariances saved
    in saved/predictions_from_nearby.
    This script constrains those posteriors to be within the measured Tn&Tx.

    Usage:
        pipeline2.jl compilestan <stan_dir>
        pipeline2.jl impute <ICAO> <model> <windownum> <data_dir> <nearby_dir> <stan_dir> [options] [--crossval]

    Options:
        --seed=<seed>  Random seed for Stan.  [default: -1]
        --ksmoothmax=<ksmoothmax>
        --epsilon=<epsilon>
        --hr_measure_true=<hr>                [default: 17]
        --hr_measure_impt=<hr>                [default: 17]
        --stan_days=<days>
"""
using DocOpt
import StanSample, MCMCChains
import Dates
using Dates: Day, Hour, Date, DateTime
using CSV
import TemperatureImputations
import PDMats, LinearAlgebra # needed for JLD.load to work, see https://github.com/JuliaIO/JLD.jl/issues/216
using DataFrames
using Printf
using Statistics
include("BatchTemperatureImputations.jl")

function parse_args()
    parse_if_not_empty(T, arg) = isnothing(arg) ? nothing : parse(T, arg)
    arguments = docopt(doc)
    stan_dir = arguments["<stan_dir>"]
    stan_dir = joinpath(stan_dir)
    if !isdir(stan_dir)
        mkpath(stan_dir)
    end
    @assert isdir(stan_dir)
    data_dir = arguments["<data_dir>"]
    data_dir = isnothing(data_dir) ? nothing : joinpath(data_dir)
    nearby_dir = arguments["<nearby_dir>"]
    windownum = parse_if_not_empty(Int, arguments["<windownum>"])
    ICAO = arguments["<ICAO>"]
    GPmodel = arguments["<model>"]
    seed = parse_if_not_empty(Int, arguments["--seed"])
    ksmoothmax = parse_if_not_empty(Float64, arguments["--ksmoothmax"])
    epsilon = parse_if_not_empty(Float64, arguments["--epsilon"])
    crossval = arguments["--crossval"]
    hr_measure_true = let arg = arguments["--hr_measure_true"]
        isnothing(arg) ? nothing : Hour(parse(Int, arg))
    end
    hr_measure_impt = let arg = arguments["--hr_measure_impt"]
        isnothing(arg) ? nothing : Hour(parse(Int, arg))
    end
    stan_days = let arg = arguments["--stan_days"]
        isnothing(arg) ? nothing : Day(parse(Int, arg))
    end
    return (stan_dir=stan_dir,
            data_dir=data_dir,
            nearby_dir=nearby_dir,
            windownum=windownum,
            ICAO=ICAO,
            GPmodel=GPmodel,
            seed=seed,
            ksmoothmax=ksmoothmax,
            epsilon=epsilon,
            crossval=crossval,
            hr_measure_true=hr_measure_true,
            hr_measure_impt=hr_measure_impt,
            stan_days=stan_days,
            compilestan=arguments["compilestan"],
            impute=arguments["impute"],
           )
end

function main()
    args = parse_args()
    imputation_model = TemperatureImputations.get_imputation_model(; pdir=args.stan_dir, seed=args.seed)
    if args.compilestan
        exit()
    end
    epsg = 5072 # doesn't actually matter in this script
    isdList = dropmissing(TemperatureImputations.read_isdList(; epsg=epsg, data_dir=args.data_dir); disallowmissing=true)
    test_station_df = isdList[isdList.ICAO .== args.ICAO, :]
    @assert nrow(test_station_df) == 1
    test_station=test_station_df[1,:]
    USAF = test_station.USAF
    WBAN = test_station.WBAN


    stan_window = BatchTemperatureImputations.imputation_chunks(;stan_days=args.stan_days)[args.windownum]
    stan_dir = args.stan_dir
    # stan_dir = BatchTemperatureImputations.stan_dirpath(;
        # save_dir=save_dir,
        # crossval=crossval,
        # GPmodel=GPmodel,
        # hr_measure=hr_measure_impt,
        # usaf=USAF,
        # wban=WBAN,
        # icao=ICAO,
        # fw=stan_window)
    # if !isdir(stan_dir)
        # mkpath(stan_dir)
    # end

    TnTx = let
        hourly_test=TemperatureImputations.read_Stations(test_station_df; data_dir=args.data_dir)
        itest=1
        TnTx_true = TemperatureImputations.test_data(hourly_test, itest, args.hr_measure_true)
        TnTx      = TemperatureImputations.test_data(hourly_test, itest, args.hr_measure_impt)
        TnTx[!,:Tn] = TnTx_true.Tn # corrupt TnTx
        TnTx[!,:Tx] = TnTx_true.Tx
        TnTx
    end
    ;

    nearby_windows = BatchTemperatureImputations.predict_from_nearby_chunks()

    stan_window_with_times = BatchTemperatureImputations.add_time_to_window(stan_window, args.hr_measure_impt)
    best_window = BatchTemperatureImputations.find_best_window(stan_window_with_times, nearby_windows)
    println("using nearby-predictions from: ", best_window)

    nearby_pred = BatchTemperatureImputations.load_predictions(args.nearby_dir, USAF, WBAN, args.ICAO, best_window)

    imputation_data, ts_window = TemperatureImputations.prep_data(
        nearby_pred,
        TnTx,
        Date(stan_window.start_date),
        args.hr_measure_impt,
        args.stan_days;
        ksmoothmax=args.ksmoothmax,
        epsilon=args.epsilon
    )


    CSV.write(joinpath(stan_dir,"timestamps.csv"), DataFrame(ts=ts_window), writeheader=false)

    @time rc = StanSample.stan_sample(imputation_model, data=imputation_data)
    @assert success(rc)

    ENV["LINES"] = 2000
    (StanSample.read_samples(imputation_model; output_format=:mcmcchains)
        |> x -> StanSample.summarystats(x)
        |> display
    )
end

main()
