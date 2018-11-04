doc = """
    * Fit the spatiotemporal GP model to nearby hourly data.
    * Generate predictions here.
    * Save the predictive mean and covariance.

    Usage:
        pipeline1.jl <ICAO> <model> <data_dir> <save_dir> [--knearest=<kn>]

    Options:
        -h --help     Show this screen.
        --knearest=<kn> Number of nearby stations to include [default: 5]
"""
using Printf
using DocOpt
using DataFrames
using JLD
using TempModel
using Dates: Date, DateTime
using GaussianProcesses: set_params!
import JSON

arguments = docopt(doc)
GPmodel = arguments["<model>"]
@show GPmodel
ICAO = arguments["<ICAO>"]
@show ICAO
data_dir= joinpath(arguments["<data_dir>"])
@show data_dir
save_dir= joinpath(arguments["<save_dir>"])
@show save_dir
k_nearest = parse(Int, arguments["--knearest"])
@show k_nearest

global k_spatiotemporal
global logNoise
if GPmodel=="fixed_var"
    k_spatiotemporal,logNoise = TempModel.fitted_sptemp_fixedvar()
elseif GPmodel=="free_var"
    k_spatiotemporal,logNoise = TempModel.fitted_sptemp_freevar()
elseif GPmodel=="sumprod"
    k_spatiotemporal,logNoise = TempModel.fitted_sptemp_sumprod()
elseif GPmodel=="SExSE"
    k_spatiotemporal,logNoise = TempModel.fitted_sptemp_SExSE()
elseif GPmodel=="diurnal"
    k_spatiotemporal,logNoise = TempModel.fitted_sptemp_diurnal()
elseif GPmodel=="simpler"
    k_spatiotemporal,logNoise = TempModel.fitted_sptemp_simpler()
elseif GPmodel=="matern"
    k_spatiotemporal,logNoise = TempModel.fitted_sptemp_matern()
else
    error(@sprintf("unknown model: %s", GPmodel))
end

# load kernel hyperparameters from JSON file
json_fname = @sprintf("hyperparams_%s_%s.json", GPmodel, ICAO) 
json_filepath = joinpath(save_dir, "fitted_kernel", GPmodel, json_fname)
open(json_filepath, "r") do io
    global output_dictionary = JSON.parse(io)
end
@assert output_dictionary["test_ICAO"] == ICAO
hyp = Float64.(output_dictionary["hyp"])
set_params!(k_spatiotemporal, hyp[2:end])
logNoise = hyp[1]

epsg = 3857 # Web Mercator (m)
isdList = TempModel.read_isdList(; data_dir=data_dir, epsg=epsg)
isd_wData = TempModel.stations_with_data(isdList; data_dir=data_dir)

test_station = isd_wData[isd_wData[:ICAO].==ICAO, :]
@assert nrow(test_station) == 1
USAF = test_station[1, :USAF]
WBAN = test_station[1, :WBAN]

isd_nearest_and_test = TempModel.find_nearest(isd_wData, USAF, WBAN, k_nearest)

@show isd_nearest_and_test

@time hourly_cat=TempModel.read_Stations(isd_nearest_and_test; data_dir=data_dir)
itest=1 # first row of isd_nearest_and_test is the test station

dt_start=DateTime(2015,1,1,0,0,0)
increm=(maximum(hourly_cat[:ts])-minimum(hourly_cat[:ts])) / 15
window=3*increm

@time while true
    global dt_start
    dt_end=dt_start+window
    savemodel_dir = joinpath(save_dir, "predictions_from_nearby", GPmodel)
    if !isdir(savemodel_dir)
        mkdir(savemodel_dir)
    end
    savemodel_dir = joinpath(savemodel_dir, ICAO)
    if !isdir(savemodel_dir)
        mkdir(savemodel_dir)
    end
    @show dt_start, dt_end
    GC.gc()
    @time nearby_pred = TempModel.predict_from_nearby(hourly_cat, isd_nearest_and_test, 
        k_spatiotemporal, logNoise,
        itest, dt_start, dt_end)
    save(joinpath(savemodel_dir,
                 @sprintf("%d_%d_%s_%s_to_%s.jld", 
                    USAF, 
                    WBAN,
                    ICAO,
                    dt_start, 
                    dt_end)), 
        "nearby_pred", 
        nearby_pred)
    if dt_end >= maximum(hourly_cat[:ts])
        break
    end
    dt_start+=increm
    GC.gc()
end
