using StatsBase: midpoints, Weights
using Base.Dates: Millisecond
function mean_temp{V1<:AbstractVector, V2<:AbstractVector}(temp::V1, ts::V2)
    temp_midpoints = midpoints(temp)
    diff_ts = diff(ts) ./ Millisecond(1)
    weights = Weights(diff_ts)
    return mean(temp_midpoints, weights)
end
