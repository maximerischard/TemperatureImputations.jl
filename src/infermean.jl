using StatsBase: midpoints, Weights
import Statistics: middle, mean
using Dates: Millisecond

function mean_temp(temp::AbstractVector, ts::AbstractVector)
    temp_midpoints = midpoints(temp)
    diff_ts = diff(ts) ./ Millisecond(1)
    weights = Weights(diff_ts)
    return mean(temp_midpoints, weights)
end

"""
    Time to the edge of the time window.
    
    For each time ts_i, returns either
        ts_i - minimum(ts)
    or
        maximum(ts) - ts_i
    whichever one is smallest.
"""
function get_buffer(ts::AbstractVector{DateTime})
    return min.(ts.-minimum(ts), maximum(ts).-ts)
end

function middle(d1::DateTime, d2::DateTime)
    diff = d2-d1
    halfdiff = diff/2
    mid = d1 + halfdiff
    return mid
end
function get_midpoints_buffer(ts::AbstractVector{DateTime})
    mid = StatsBase.midpoints(ts)
    return min.(mid.-minimum(ts), maximum(ts).-mid)
end

# function extract_mean(chains::AxisArray, ts::AbstractVector{DateTime}, window_num::Int, best_df::DataFrame)
    # # extract imputed temperatures
    # temp_impute = get_temperatures_reparam(chains)
    # is_best = best_df[:window] .== window_num
    # imid_is_best = best_df[is_best, :imid]
    # best_istart, best_iend = extrema(imid_is_best)
    # ts_best = ts[best_istart:best_iend+1]
    # println("obtaining mean between ", ts[best_istart], " and ", ts[best_iend+1])
    # nchains = size(temp_impute, :chain)
    # nsamples = size(temp_impute, :sample)
    # mean_by_sample = Vector{Float64}(nchains*nsamples)
    # isample_x_chain = 0
    # for ichain in 1:nchains
        # for isample in 1:nsamples
            # isample_x_chain += 1

            # temp = view(temp_impute, 
                # Axis{:sample}(isample), 
                # Axis{:param}(:), 
                # Axis{:chain}(ichain))
            # temp_best = view(temp, best_istart:best_iend+1)
            # @assert length(temp_best) == length(ts_best)
            # mean_best = mean_temp(temp_best, ts_best)

            # mean_by_sample[isample_x_chain] = mean_best
        # end
    # end
    # duration = ts[best_iend+1]-ts[best_istart]
    # return mean_by_sample, duration
# end
"""
    Given Stan samples, extract the mean temperature, applied over the subset
    of times for which this set of imputations is most reliable (away from the edges).
"""
function get_means_by_day(chains, ts, hr_measure)
    nsamples, _, nchains = size(chains)
    temp_impute = get_temperatures_reparam(chains)
    ts_mid, ts_diff = midpoints(ts), diff(ts)
    ts_mid_day = TempModel.measurement_date.(ts_mid, hr_measure)
    nsamples, ntimes, nchains = size(chains)
    days = unique(ts_mid_day)
    ndays = length(days)
    means_by_day = Array{Float64}(undef, nsamples, nchains, ndays)#Dict{eltype(days), Matrix{Float64}}()
    for iday in 1:ndays
       day = days[iday]
       # day_means = Matrix{Float64}(undef, nsamples, nchains)
       # means_by_day[day] = day_means
       sub = ts_mid_day .== day
       ts_sub, diff_sub = ts_mid[sub], ts_diff[sub]
       weights = Weights(diff_sub ./ Millisecond(1))
       for ichain in 1:nchains
           for isample in 1:nsamples
               temp_sub = midpoints(@view(temp_impute[isample, :, ichain]))[sub]
               weighted_mean = mean(temp_sub, weights)
               means_by_day[isample, ichain, iday] = weighted_mean
            end
        end
    end
    return days, means_by_day
end
