doc = """
    * Fit the spatiotemporal GP model to nearby hourly data.
    * Generate predictions here.
    * Save the predictive mean and covariance.

    Usage:
        batch_infermean.jl <model> <data_dir> <save_dir> [<ICAO>] [--crossval]

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
@show GPmodel
ICAO = arguments["<ICAO>"]
@show ICAO
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
@show data_dir
save_dir= joinpath(arguments["<save_dir>"])
@show save_dir
@assert isdir(save_dir)
crossval = arguments["--crossval"]::Bool
@show crossval


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

function true_mean_day(hourly, hr_measure)
    ts, temp = hourly[:ts], hourly[:temp]
    temp_mid = midpoints(temp)
    ts_mid, ts_diff = midpoints(ts), diff(ts)
    ts_mid_day = TempModel.measurement_date.(ts_mid, hr_measure)
    days = sort(unique(ts_mid_day))[1:end]
    ndays = length(days)
    means_by_day = Float64[]
    total_duration = Float64[]
    for iday in 1:ndays
        day = days[iday]
        sub = ts_mid_day .== day
        ts_sub, diff_sub = ts_mid[sub], ts_diff[sub]
        weights = Weights(diff_sub ./ Millisecond(1))
        temp_sub = temp_mid[sub]
        mean_temp = mean(temp_sub, weights)
        push!(means_by_day, mean_temp)
        push!(total_duration, sum(diff_sub) ./ convert(Millisecond, Hour(1)))
    end
    return days, means_by_day, total_duration
end

epsg = 3857 # Web Mercator (m)
isdList = TempModel.read_isdList(; data_dir=data_dir, epsg=epsg)
isd_wData = TempModel.stations_with_data(isdList; data_dir=data_dir)

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
    hourly_test[:ts_day] = [TempModel.measurement_date(t, hr_measure) for t in hourly_test[:ts]]
    TnTx = DataFrames.by(hourly_test, :ts_day, df -> DataFrame(
        Tn=minimum(df[:temp]), 
        Tx=maximum(df[:temp])))
    # add column to test data for TnTx (useful for plotting)
    test_trimmed=join(hourly_test, TnTx, on=:ts_day)

    days, true_means_by_day, total_duration = true_mean_day(hourly_test, hr_measure)
    @show ICAO
    @show mean(true_means_by_day, Weights(total_duration))
    ;

    out_save_dir = joinpath(save_dir, "daily_mean", crossval ? "crossval" : "mll", GPmodel)
    mkpath(out_save_dir)
    filepath = joinpath(out_save_dir, "daily_means_$(ICAO).json")
    try
        open(filepath, "r") do io
            global all_posteriors = JSON.parse(io)
        end
    catch
        println("can't open: ", filepath)
        continue
    end
        
    days_vec, day_means, day_cov, day_buffer, buffer_cov = Batch.daily_best(all_posteriors);
    @show days[1:10]
    @show days_vec[1:10]
    @assert days[1:length(days_vec)] == days_vec

    total_duration_overlap = total_duration[1:length(day_means)]
    yearly_mean = sum(day_means .* total_duration_overlap) / sum(total_duration_overlap)
    @printf("yearly posterior mean: %6.3f\n", yearly_mean)
    yearly_var = (total_duration_overlap'*day_cov*total_duration_overlap) / sum(total_duration_overlap)^2
    yearly_std = √(yearly_var)
    @printf("yearly posterior std:  %6.3f\n", √yearly_var)

    yearly_true_mean = mean(true_means_by_day[1:length(day_means)], Weights(total_duration_overlap))
    @show yearly_true_mean

    sigma = (yearly_mean - yearly_true_mean) / yearly_std
    @show sigma

    # systematic_cov(kij) = kij == 0 ? 0.01 : kij
    # systematic_std = sqrt(
        # (total_duration_overlap' 
         # * systematic_cov.(day_cov)
         # *total_duration_overlap) 
        # / sum(total_duration_overlap)^2
    # )
    # @printf("yearly posterior std with systematic error 0.01: %.3f\n", systematic_std)
    # sigma_syst = (yearly_mean - yearly_true_mean) / systematic_std
    # @show sigma_syst
    println(yearly_true_mean, "\t", yearly_mean, "\t", yearly_std, "\t", sigma)
end
