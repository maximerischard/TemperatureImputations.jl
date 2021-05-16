doc = """
    * Fit the hyperparameters of the spatiotemporal GP model to nearby hourly data.

    Usage:
        pipeline0_kernelfit.jl <ICAO> <model> <data_dir> <save_dir> --timelimit=<timelimit> [--knearest=<kn>] [--crossval]

    Options:
        -h --help        Show this screen.
        --knearest=<kn>  Number of nearby stations to include [default: 5]
        --crossval       Use cross-validation
"""
using DocOpt
import TemperatureImputations
using Printf: @printf, @sprintf
import JSON
using DataFrames: nrow
using Dates: Day

arguments = docopt(doc)
ICAO = arguments["<ICAO>"]
@show ICAO
GPmodel = arguments["<model>"]
@show GPmodel
data_dir= joinpath(arguments["<data_dir>"])
@show data_dir
save_dir= joinpath(arguments["<save_dir>"])
@show save_dir
@assert isdir(save_dir)
k_nearest = parse(Int, arguments["--knearest"])
@show k_nearest
crossval = arguments["--crossval"]::Bool
@show crossval
timelimit = parse(Float64, arguments["--timelimit"])
@show timelimit

global k_spatiotemporal
global logNoise
if GPmodel=="fixed_var"
    k_spatiotemporal,logNoise = TemperatureImputations.fitted_sptemp_fixedvar(;kmean=true)
elseif GPmodel=="free_var"
    k_spatiotemporal,logNoise = TemperatureImputations.fitted_sptemp_freevar(;kmean=true)
elseif GPmodel=="sumprod"
    k_spatiotemporal,logNoise = TemperatureImputations.fitted_sptemp_sumprod(;kmean=true)
elseif GPmodel=="SExSE"
    kdict = TemperatureImputations.kernel_sptemp_SExSE(;kmean=true)
    k_spatiotemporal = kdict[:spatiotemporal]
    logNoise = -1.0
elseif GPmodel=="diurnal"
    k_spatiotemporal,logNoise = TemperatureImputations.fitted_sptemp_diurnal(;kmean=true)
elseif GPmodel=="simpler"
    k_spatiotemporal,logNoise = TemperatureImputations.fitted_sptemp_simpler(;kmean=true)
elseif GPmodel=="matern"
    kdict = TemperatureImputations.kernel_sptemp_matern(;kmean=true)
    k_spatiotemporal = kdict[:spatiotemporal]
    logNoise = -1.0
elseif GPmodel=="maternlocal"
    kdict = TemperatureImputations.kernel_sptemp_maternlocal(;kmean=true)
    k_spatiotemporal = kdict[:spatiotemporal]
    logNoise = -1.0
else
    error(@sprintf("unknown model: %s", GPmodel))
end

epsg = 3857 # Web Mercator (m)
isdList = TemperatureImputations.read_isdList(; data_dir=data_dir, epsg=epsg)
isd_wData = TemperatureImputations.stations_with_data(isdList; data_dir=data_dir)

test_station = isd_wData[isd_wData.ICAO.==ICAO, :]
@assert nrow(test_station) == 1
USAF = test_station[1, :USAF]
WBAN = test_station[1, :WBAN]

isd_nearest_and_test = TemperatureImputations.find_nearest(isd_wData, USAF, WBAN, k_nearest)
isd_nearest = isd_nearest_and_test[2:end,:]

hourly_data = TemperatureImputations.read_Stations(isd_nearest; data_dir=data_dir)

if !crossval
    @time opt_out = TemperatureImputations.optim_kernel(k_spatiotemporal, logNoise, isd_nearest, hourly_data, :Optim; 
                                           x_tol=1e-4, f_tol=1e-8,
                                           time_limit=timelimit,
                                           show_trace=true,
                                           show_every=5,
                                           window=Day(10));
    hyp = opt_out[:hyp]
    output_dictionary = Dict{String,Any}(
        "mll" => -opt_out[:minimum],
        "hyp" => opt_out[:hyp],
        "logNoise" => opt_out[:logNoise],
        "test_ICAO" => ICAO,
        "test_USAF" => USAF,
        "test_WBAN" => WBAN,
        "nearby_ICAO" => isd_nearest.ICAO,
        "nearby_USAF" => isd_nearest.USAF,
        "nearby_WBAN" => isd_nearest.WBAN,
        "GPmodel" => GPmodel
       )
else
    @time opt_out_CV = TemperatureImputations.optim_kernel_CV(k_spatiotemporal, -2.0, 
                                           isd_nearest, hourly_data,
                                           :Optim;
                                           window=Day(8), # shorter window is faster
                                           x_tol=1e-4, f_tol=1e-6,
                                           time_limit=timelimit,
                                           show_trace=true,
                                           show_every=5,
                                           )
    hypCV = opt_out_CV[:hyp]
    @show opt_out_CV[:opt_out]
    @show hypCV
    # reals, folds = TemperatureImputations.make_chunks_and_folds(k_spatiotemporal, hypCV[1], isd_nearest, 
            # hourly_data; window=Day(10));
    # update_mll!(reals)
    # mll = reals.mll
    output_dictionary = Dict{String,Any}(
        "CV" => -opt_out_CV[:minimum],
        "hyp" => opt_out_CV[:hyp],
        "logNoise" => opt_out_CV[:logNoise],
        "test_ICAO" => ICAO,
        "test_USAF" => USAF,
        "test_WBAN" => WBAN,
        "nearby_ICAO" => isd_nearest.ICAO,
        "nearby_USAF" => isd_nearest.USAF,
        "nearby_WBAN" => isd_nearest.WBAN,
        "GPmodel" => GPmodel
       )
end

if crossval
    savemodel_dir = joinpath(save_dir, "fitted_kernel", "crossval", GPmodel)
else
    savemodel_dir = joinpath(save_dir, "fitted_kernel", "mll", GPmodel)
end
if !isdir(savemodel_dir)
    mkpath(savemodel_dir)
end
fname = @sprintf("hyperparams_%s_%s.json", GPmodel, ICAO) 
filepath = joinpath(savemodel_dir, fname)
open(filepath, "w") do io
    indent = 4
    JSON.print(io, output_dictionary, indent)
end
