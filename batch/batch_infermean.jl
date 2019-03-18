doc = """
    * Fit the spatiotemporal GP model to nearby hourly data.
    * Generate predictions here.
    * Save the predictive mean and covariance.

    Usage:
        batch_infermean.jl <model> <data_dir> <save_dir> [<ICAO>] [--crossval] [--verbose]

    Options:
        -h --help     Show this screen.
"""
using DocOpt
using DataFrames
using TempModel
using Dates
using PDMats
using LinearAlgebra
using Printf
using Statistics
using StatsBase
using JSON

arguments = docopt(doc)
GPmodel = arguments["<model>"]
verbose = arguments["--verbose"]
ICAO = arguments["<ICAO>"]
if ICAO == nothing
    ICAO_list= ["KABE", "KABQ", "KABR", "KATL", "KAUG", "KBDL", "KBHM", "KBIS", 
                "KBNA", "KBWI", "KCAE", "KCEF", "KCMH", "KCOS", "KCRW", "KDLH", 
                "KDSM", "KEUG", "KFAT", "KFYV", "KGTF", "KICT", "KIND", "KINW", 
                "KJAN", "KJAX", "KLBF", "KLEX", "KLSE", "KMPV", "KMWL", "KOKC", 
                "KPIH", "KPLN", "KPVD", "KRDU", "KROA", "KSEA", "KSGF", "KSHV", 
                "KSLC", "KSPI", "KSYR", "KTPH"]
else
    ICAO_list = [ICAO]
end
data_dir= joinpath(arguments["<data_dir>"])
save_dir= joinpath(arguments["<save_dir>"])
crossval = arguments["--crossval"]::Bool
@assert isdir(save_dir)
if verbose
    @show data_dir
    @show save_dir
    @show crossval
    @show GPmodel
end


module Batch
    using ..TempModel
    using DataFrames
    using Dates
    using Distributions
    using PDMats
    using LinearAlgebra
    using Printf
    ;
    src_dir = dirname(pathof(TempModel))
    include(joinpath(src_dir, "batch_diagnostics.jl"))
    include(joinpath(src_dir, "infermean.jl"))
end


function get_day_durations(ts_min, ts_max, hr_measure::Hour)
    first_measurement = DateTime(Date(ts_min)) + hr_measure
    if first_measurement < ts_min
        first_measurement += Day(1)
    end
    measurement_times = collect(first_measurement:Day(1):ts_max)
    measurement_edges = [ts_min; measurement_times; ts_max]
    durations_hours = diff(measurement_edges) ./ (convert(Millisecond, Hour(1)))
    day_edges = TempModel.measurement_date.(measurement_edges, hr_measure)
    @assert measurement_times[1] != ts_min # would need to think about this
    @assert measurement_times[end] != ts_max # would need to think about this
    # if measurement_times[1] == ts_min
        # @show hr_measure
        # @show ts_min
        # @show duration_hours[1:5]
        # @show measurement_edges[1:5]
    # end
    return durations_hours
    # if measurement_times[end] == ts_max
        # return durations_hours[1:end-1]
    # else
        # return durations_hours
    # end
end

function get_meanTnTx(TnTx::DataFrame, hr_measure::Hour, ts_min::DateTime, ts_max::DateTime)
    durations = get_day_durations(ts_min, ts_max, hr_measure)
    # @show hr_measure, ts_min, ts_max, nrow(TnTx), length(durations)
    total_duration = sum(durations)
    return (mean(TnTx[:Tn], Weights(durations)) +
            mean(TnTx[:Tx], Weights(durations))
           ) / 2
end

function true_mean_day(hourly, hr_measure)
    ts, temp = hourly[:ts], hourly[:temp]
    temp_mid = midpoints(temp)
    ts_mid, ts_diff = midpoints(ts), diff(ts)
    ts_mid_day = TempModel.measurement_date.(ts_mid, hr_measure)
    days = sort(unique(ts_mid_day))[1:end]
    ndays = length(days)
    means_by_day = Float64[]
    day_duration = Float64[]
    for iday in 1:ndays
        day = days[iday]
        sub = ts_mid_day .== day
        ts_sub, diff_sub = ts_mid[sub], ts_diff[sub]
        weights = Weights(diff_sub ./ Millisecond(1))
        temp_sub = temp_mid[sub]
        mean_temp = mean(temp_sub, weights)
        push!(means_by_day, mean_temp)
        push!(day_duration, sum(diff_sub) ./ convert(Millisecond, Hour(1)))
    end
    return days, means_by_day, day_duration
end

epsg = 3857 # Web Mercator (m)
isdList = TempModel.read_isdList(; data_dir=data_dir, epsg=epsg)
isd_wData = TempModel.stations_with_data(isdList; data_dir=data_dir)

println(join(("ICAO", "true", "meanTnTx17", "meanTnTx_min", "meanTnTx_max", "imputed_mean", "imputed_std", "sigma", "augmented_std"), ","))
for ICAO in ICAO_list
    test_station = isd_wData[isd_wData[:ICAO].==ICAO, :]
    @assert nrow(test_station) == 1
    USAF = test_station[1, :USAF]
    WBAN = test_station[1, :WBAN]

    k_nearest = 5
    isd_nearest_and_test = TempModel.find_nearest(isd_wData, USAF, WBAN, k_nearest)

    timezoneGMT = -5 # Georgia
    # timezoneGMT = -7 # Arizona
    local_time(ts) = ts - Hour(timezoneGMT)
    ilocaltime = circshift(1:24, -timezoneGMT)
    ;

    # obtain the hourly temperature measurements for those stations
    hourly_cat=TempModel.read_Stations(isd_nearest_and_test; data_dir=data_dir)
    # mark station 1 as the test station
    itest=1
    # separate temperatures into train and test
    hourly_train = hourly_cat[hourly_cat[:station].!=itest,:]
    hourly_test  = hourly_cat[hourly_cat[:station].==itest,:]

    # emulate daily Tx/Tn measurement
    hr_measure = Hour(17) # number between 0 and 24
    TnTx = TempModel.test_data(hourly_test, itest, hr_measure)

    true_days, true_means_by_day, day_duration = true_mean_day(hourly_test, hr_measure)
    if verbose
        @show mean(true_means_by_day, Weights(day_duration))
    end
    ;


    out_save_dir = joinpath(save_dir, "daily_mean", crossval ? "crossval" : "mll", GPmodel)
    mkpath(out_save_dir)
    filepath = joinpath(out_save_dir, "daily_means_$(ICAO).json")
    try
        open(filepath, "r") do io
            global all_posteriors = JSON.parse(io)
        end
    catch
        if verbose
            println("can't open: ", filepath)
        end
        continue
    end
        
    # if ICAO ∈ ["KBDL",]
        # continue
    # end
    if verbose
        @show ICAO
    end
    imput_days, imput_day_means, day_cov, day_buffer, buffer_cov = Batch.daily_best(all_posteriors);
    withimputations = 1:length(imput_days)
    @assert true_days[withimputations] == imput_days


    @assert length(imput_days) == length(true_days) # actually...
    ts_min, ts_max = extrema(hourly_test[:ts])
    meanTnTx = get_meanTnTx(TnTx, hr_measure, ts_min, ts_max)
    meanTnTx_by_hr = [get_meanTnTx(TempModel.test_data(hourly_test, itest, hr), hr, ts_min, ts_max)
                      for hr in Hour(1):Hour(1):Hour(24)]
    meanTnTx_min, meanTnTx_max = extrema(meanTnTx_by_hr)
    if verbose
        @show meanTnTx, meanTnTx_min, meanTnTx_max
    end

    day_duration_overlap = day_duration[withimputations]
    yearly_mean = sum(imput_day_means .* day_duration_overlap) / sum(day_duration_overlap)
    if verbose
        @printf("yearly posterior mean: %6.3f\n", yearly_mean)
    end
    yearly_var = (day_duration_overlap'*day_cov*day_duration_overlap) / sum(day_duration_overlap)^2
    yearly_std = √(yearly_var)
    if verbose
        @printf("yearly posterior std:  %6.3f\n", √yearly_var)
    end

    yearly_true_mean = mean(true_means_by_day[withimputations], Weights(day_duration_overlap))
    if verbose
        @show yearly_true_mean
    end

    sigma = (yearly_mean - yearly_true_mean) / yearly_std
    if verbose
        @show sigma
    end

    meancovlag = Float64[]
    numlag = Int[]
    for lag in 1:size(day_cov,1)
        nonzero_offdiags = [c for c in diag(day_cov, lag) if c!=0]
        if length(nonzero_offdiags) == 0
            break
        end
        meancov = mean(nonzero_offdiags)
        num = length(nonzero_offdiags)
        if verbose
            @printf("lag %2d (%3d obs): %.4f\n", lag, num, meancov)
        end
        push!(meancovlag, meancov)
        push!(numlag, num)
    end
    meancovgt5 = max(0.0, mean(meancovlag[5:end], Weights(numlag[5:end])))
    atadistance(x, σ) = (x==0 ? σ : x)
    augmented_cov = atadistance.(day_cov, meancovgt5)
    augmented_std = sqrt(day_duration_overlap'*augmented_cov*day_duration_overlap 
                         / sum(day_duration_overlap)^2)

    if verbose
        @show augmented_std
    end

    println(ICAO, ",", yearly_true_mean, ",", meanTnTx, ",", meanTnTx_min, ",", meanTnTx_max, ",", yearly_mean, ",", yearly_std, ",", sigma, ",", augmented_std)
end
