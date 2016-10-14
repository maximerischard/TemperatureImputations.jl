
using DataFrames
using GaussianProcesses
using Proj4
using GaussianProcesses: set_params!
;

using JLD

include("src/preprocessing.jl")
include("src/variogram.jl")

isdList=read_isdList()
isdSubset=isdList[[(usaf in (725450,725460,725480,725485)) for usaf in isdList[:USAF].values],:]
isdSubset

hourly_cat=read_Stations(isdSubset)
itest=3

include("src/fitted_kernel.jl")

module pred
    include("src/predict_from_nearby.jl")
end

test_usaf=get(isdSubset[itest,:USAF])

dt_start=DateTime(2015,1,1,0,0,0)
increm=get(maximum(hourly_cat[:ts])-minimum(hourly_cat[:ts])) / 15
window=3*increm

while true
    dt_end=dt_start+window
    nearby_pred = pred.predict_from_nearby(hourly_cat, isdSubset, 
        k_spatiotemporal, logNoise,
        itest, dt_start, dt_end)
    save(@sprintf("saved/predictions_from_nearby/%d_%s_to_%s.jld", test_usaf, Date(dt_start), Date(dt_end)), 
        "nearby_pred", 
        nearby_pred)
    if dt_end >= get(maximum(hourly_cat[:ts]))
        break
    end
    dt_start+=increm
end
