doc = """
    * Fit the spatiotemporal GP model to nearby hourly data.
    * Generate predictions here.
    * Save the predictive mean and covariance.

    Usage:
        pipeline1.jl <model> <data_dir> <save_dir>
"""
using DocOpt
using DataFrames
using JLD
using TempModel
using Dates: Date

arguments = docopt(doc)
GPmodel = arguments["<model>"]
data_dir= arguments["<data_dir>"]
save_dir= arguments["<save_dir>"]
save_dir = joinpath(save_dir)
println("directory for saved files: ", save_dir)

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

isdList=TempModel.read_isdList(; data_dir=data_dir)
isdSubset=isdList[[(usaf in (725450,725460,725480,725485)) for usaf in isdList[:USAF]],:]
isdSubset

hourly_cat=TempModel.read_Stations(isdSubset; data_dir=data_dir)
itest=3

test_usaf=isdSubset[itest,:USAF]

dt_start=DateTime(2015,1,1,0,0,0)
increm=(maximum(hourly_cat[:ts])-minimum(hourly_cat[:ts])) / 15
window=3*increm

while true
    global dt_start
    dt_end=dt_start+window
    savemodel_dir = joinpath(save_dir, "predictions_from_nearby", GPmodel)
    if !isdir(savemodel_dir)
        mkdir(savemodel_dir)
    end
    nearby_pred = TempModel.predict_from_nearby(hourly_cat, isdSubset, 
        k_spatiotemporal, logNoise,
        itest, dt_start, dt_end)
    save(joinpath(savemodel_dir,
                 @sprintf("%d_%s_to_%s.jld", 
                    test_usaf, 
                    Date(dt_start), 
                    Date(dt_end))), 
        "nearby_pred", 
        nearby_pred)
    if dt_end >= maximum(hourly_cat[:ts])
        break
    end
    dt_start+=increm
end
