# Empirical variograms
mutable struct Variogram
    bins::AbstractVector
    sqdiff_accum::Vector{Float64}
    npairs::Vector{Int}
    function Variogram(bins::AbstractVector)
        nbins = length(bins)
        sqdiff_accum = zeros(Float64, nbins)
        npairs = zeros(Int, nbins)
        return new(bins, sqdiff_accum, npairs)
    end
end 
semivariog(v::Variogram) = v.sqdiff_accum ./ (v.npairs.*2)

function timeseries_variogram(times, values, bins)
    v = Variogram(bins)
    N = length(times)
    @assert length(values) == N
    for i in 1:N
        t1 = times[i]
        ibin = 1
        for j in i+1:N
            Δt = times[j] - t1
            while Δt>bins[ibin]
                ibin += 1
            end
            v.npairs[ibin] += 1
            v.sqdiff_accum[ibin] += (values[j]-values[i])^2
        end
    end
    return v
end

function cov(sumkern::SumKernel, r::Float64)
    ck = 0.0
    for k in sumkern.kerns
        metr = evaluate(metric(k), [r], [0.0])
        ck += cov(k, metr)
    end
    return ck
end

"""
    Empirical variogram between two time series.
"""
function cross_variog(times_i, times_j, temp_i, temp_j, timebins)
    v = Variogram(timebins)
    temp_i = temp_i.-mean(temp_i)
    temp_j = temp_j.-mean(temp_j)
    nbins = length(v.bins)::Int64
    @inbounds for i in 1:length(times_i)
        t_i = times_i[i]
        y_i = temp_i[i]
        for j in 1:length(times_j)
            t_j = times_j[j]
            y_j = temp_j[j]
            timediff = abs(t_j-t_i)
            ibin = searchsortedlast(timebins, timediff)
            if ibin == nbins
                continue
            end
            v.sqdiff_accum[ibin] += (y_i-y_j)^2
            v.npairs[ibin] += 1
        end
    end
    return v
end
