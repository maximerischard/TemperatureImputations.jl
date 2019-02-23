doc = """
    * Fit the hyperparameters of the spatiotemporal GP model to nearby hourly data.

    Usage:
        pipeline0_kernelfit.jl <ICAO> <model> <data_dir> <save_dir> [--knearest=<kn>] [--crossval]

    Options:
        -h --help        Show this screen.
        --knearest=<kn>  Number of nearby stations to include [default: 5]
        --crossval       Use cross-validation
"""
using DocOpt
import TempModel
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
    kdict = TempModel.kernel_sptemp_matern(;kmean=true)
    k_spatiotemporal = kdict[:spatiotemporal]
    logNoise = -1.0
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

if !crossval
    @time opt_out = TempModel.optim_kernel(k_spatiotemporal, logNoise, isd_nearest, hourly_data, :Optim; window=Day(10));
    hyp = opt_out[:hyp]
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
else
    @time opt_out = TempModel.optim_kernel(k_spatiotemporal, logNoise, isd_nearest, hourly_data, :Optim; window=Day(8), x_tol=1e-5, f_tol=1e-5);
    hyp = opt_out[:hyp]
    opt_out_CV = TempModel.optim_kernel_CV(k_spatiotemporal, hyp[1], 
                                           isd_nearest, hourly_data,
                                           :Optim;
                                           window=Day(8), # shorter window is faster
                                           )
    hypCV = opt_out_CV[:hyp]
    reals, folds = make_chunks_and_folds(k_spatiotemporal, hypCV[1], isd_nearest, 
            hourly_data; window=Day(10));
    mll = reals.mll
    output_dictionary = Dict{String,Any}(
        "mll" => mll,
        "CV" => opt_out_CV[:mll],
        "hyp" => opt_out_CV[:hyp],
        "logNoise" => opt_out_CV[:logNoise],
        "test_ICAO" => ICAO,
        "test_USAF" => USAF,
        "test_WBAN" => WBAN,
        "nearby_ICAO" => isd_nearest[:ICAO],
        "nearby_USAF" => isd_nearest[:USAF],
        "nearby_WBAN" => isd_nearest[:WBAN],
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

# the code below is just to nicely print the results
# all the required output information is already
# saved in the json file above

ksum, k_st = k_spatiotemporal.kleft, k_spatiotemporal.kright
ksum, k_st = ksum.kleft, ksum.kright
k4, ksp4 = k_st.kleft, k_st.kright
k4 = k4.kernel
ksp4 = ksp4.kernel.kernel

ksum, k_st = ksum.kleft, ksum.kright
k3, ksp3 = k_st.kleft, k_st.kright
k3 = k3.kernel
ksp3 = ksp3.kernel.kernel

ksum, k_st = ksum.kleft, ksum.kright
k2, ksp2 = k_st.kleft, k_st.kright
k2 = k2.kernel
ksp2 = ksp2.kernel.kernel

k_st = ksum
k1, ksp1 = k_st.kleft, k_st.kright
k1 = k1.kernel.kernel
ksp1 = ksp1.kernel.kernel

print("k₁: Periodic \n=================\n")
@printf("σ: %5.3f\n", √k1.σ2)
@printf("l: %5.3f\n", √k1.ℓ2)
@printf("p: %5.0f hours\n", k1.p)
print("> spatial decay:\n")
@printf("l: %5.3f km\n", ksp1.ℓ / 1000)
print("\nk₂: RQIso \n=================\n")
@printf("σ: %5.3f\n", √k2.σ2)
@printf("l: %5.3f hours\n", √ k2.ℓ2)
@printf("α: %5.3f\n", k2.α)
print("> spatial decay:\n")
# @printf("σ: %5.3f\n", √ksp2.σ2)
@printf("l: %5.3f km\n", ksp2.ℓ / 1000)
print("\nk₃: SEIso \n=================\n")
@printf("σ: %5.3f\n", √k3.σ2)
@printf("l: %5.3f hours\n", √k3.ℓ2)
print("> spatial decay:\n")
# @printf("σ: %5.3f\n", √ksp3.σ2)
@printf("l: %5.3f km\n", ksp3.ℓ / 1000)
print("\nk₄: RQIso \n=================\n")
@printf("σ: %5.3f\n", √k4.σ2)
@printf("l: %5.3f days\n", √k4.ℓ2 / 24)
@printf("α: %5.3f\n",  k4.α)
print("> spatial decay:\n")
# @printf("σ: %5.3f\n", √ksp4.σ2)
@printf("l: %5.3f km\n", ksp4.ℓ / 1000)
print("\n=================\n")
@printf("σy: %5.3f\n", exp(opt_out[:hyp][1]))
