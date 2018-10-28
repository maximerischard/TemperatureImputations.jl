doc = """
    * Fit the hyperparameters of the spatiotemporal GP model to nearby hourly data.

    Usage:
        pipeline0_kernelfit.jl <ICAO> <model> <data_dir> <save_dir> [--knearest=<kn>]

    Options:
        -h --help     Show this screen.
        --knearest=<kn> Number of nearby stations to include [default: 5]
"""
using DocOpt
import TempModel
using Printf: @printf, @sprintf
import JSON

arguments = docopt(doc)
ICAO = arguments["<ICAO>"]
@show ICAO
GPmodel = arguments["<model>"]
@show GPmodel
data_dir= arguments["<data_dir>"]
@show data_dir
save_dir= arguments["<save_dir>"]
@show save_dir
k_nearest = parseInt(arguments["<kn>"])
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

epsg = 3857 # Web Mercator (m)
isdList = TempModel.read_isdList(; data_dir=data_dir, epsg=epsg)
isd_wData = TempModel.stations_with_data(isdList; data_dir=data_dir)

test_station = isd_wData[isd_wData[:ICAO].==ICAO, :]
@assert nrow(test_station) == 1
USAF = test_station[1, :USAF]
WBAN = test_station[1, :WBAN]

isd_nearest_and_test = TempModel.find_nearest(isd_wData, USAF, WBAN, k_nearest)
isd_nearest = isd_nearest_and_test[2:end,:]

hourly_data = TempModel.read_Stations(isd_nearest; data_dir=data_dir)

@time opt_out = TempModel.optim_kernel(k_spatiotemporal, logNoise, isd_nearest, hourly_data, :Optim);

output_dictionary = Dict{String,Any}(
    "mll" => opt_out[:mll],
    "hyp" => opt_out[:hyp],
    "logNoise" => opt_out[:logNoise],
    "test_ICAO" => ICAO,
    "test_USAF" => USAF,
    "test_WBAN" => WBAN,
    "nearby_ICAO" => isd_nearest[:ICAO],
    "nearby_USAF" => isd_nearest[:USAF],
    "nearby_WBAN" => isd_nearest[:WBAN],
    "GPmodel" => GPmodel
   )

savemodel_dir = joinpath(save_dir, "fitted_kernel", GPmodel)
if !isdir(savemodel_dir)
    mkdir(savemodel_dir)
end
fname = @sprintf("hyperparams_%s_%s.json", GPmodel, ICAO) 
filepath = joinpath(savemodel_dir, fname)
open(filepath, "w") do io
    indent = 4
    JSON.print(io, output_dictionary, indent)
end

print("k₁: Periodic \n=================\n")
@printf("σ: %5.3f\n", √k1.kernel.σ2)
@printf("l: %5.3f\n", √k1.kernel.ℓ2)
@printf("p: %5.0f hours\n", k1.kernel.p)
print("> spatial decay:\n")
@printf("l: %5.3f km\n", ksp1.kernel.ℓ / 1000)
print("\nk₂: RQIso \n=================\n")
@printf("σ: %5.3f\n", √k2.σ2)
@printf("l: %5.3f hours\n", √ k2.ℓ2)
@printf("α: %5.3f\n", k2.α)
print("> spatial decay:\n")
# @printf("σ: %5.3f\n", √ksp2.σ2)
@printf("l: %5.3f km\n", ksp2.kernel.ℓ / 1000)
print("\nk₃: SEIso \n=================\n")
@printf("σ: %5.3f\n", √k3.σ2)
@printf("l: %5.3f hours\n", √k3.ℓ2)
print("> spatial decay:\n")
# @printf("σ: %5.3f\n", √ksp3.σ2)
@printf("l: %5.3f km\n", ksp3.kernel.ℓ / 1000)
print("\nk₄: RQIso \n=================\n")
@printf("σ: %5.3f\n", √k4.σ2)
@printf("l: %5.3f days\n", √k4.ℓ2 / 24)
@printf("α: %5.3f\n",  k4.α)
print("> spatial decay:\n")
# @printf("σ: %5.3f\n", √ksp4.σ2)
@printf("l: %5.3f km\n", ksp4.kernel.ℓ / 1000)
print("\n=================\n")
@printf("σy: %5.3f\n", exp(opt_out[:hyp][1]))
