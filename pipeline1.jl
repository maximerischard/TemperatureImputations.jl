doc = """
    * Fit the spatiotemporal GP model to nearby hourly data.
    * Generate predictions here.
    * Save the predictive mean and covariance.

    Usage:
        pipeline1.jl <model>
"""
using DocOpt
arguments = docopt(doc)
GPmodel = arguments["<model>"]
module TempModel
    using TimeSeries
    using DataFrames
    using GaussianProcesses
    using Proj4
    using GaussianProcesses: set_params!
    using PDMats
    using GaussianProcesses
    include("src/predict_from_nearby.jl")
    include("src/preprocessing.jl")
    include("src/variogram.jl")
    include("src/fitted_kernel.jl")
end

using JLD

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
else
    error(@sprintf("unknown model: %s", GPmodel))
end

isdList=TempModel.read_isdList()
isdSubset=isdList[[(usaf in (725450,725460,725480,725485)) for usaf in isdList[:USAF].values],:]
isdSubset

hourly_cat=TempModel.read_Stations(isdSubset)
itest=3

test_usaf=get(isdSubset[itest,:USAF])

dt_start=DateTime(2015,1,1,0,0,0)
increm=get(maximum(hourly_cat[:ts])-minimum(hourly_cat[:ts])) / 15
window=3*increm

while true
    dt_end=dt_start+window
    saved_dir = joinpath(pwd(), "saved/predictions_from_nearby", GPmodel)
    if !isdir(saved_dir)
        mkdir(saved_dir)
    end
    nearby_pred = TempModel.predict_from_nearby(hourly_cat, isdSubset, 
        k_spatiotemporal, logNoise,
        itest, dt_start, dt_end)
    save(joinpath(saved_dir,
                 @sprintf("%d_%s_to_%s.jld", 
                    test_usaf, 
                    Date(dt_start), 
                    Date(dt_end))), 
        "nearby_pred", 
        nearby_pred)
    if dt_end >= get(maximum(hourly_cat[:ts]))
        break
    end
    dt_start+=increm
end
