using LinearAlgebra
using PDMats
import LogDensityProblems
import LogDensityProblems: dimension, logdensity
using LogDensityProblems: Value, ValueGradient
struct PredictTemperatures <: LogDensityProblems.AbstractLogDensityProblem
    impt_times_p_day::Vector{Int}
    Tn::Vector{Float64}
    Tx::Vector{Float64}
    μ::Vector{Float64}
    chol::LowerTriangular{Float64,Array{Float64,2}}
    Nimpt::Int
    day_impute::Vector{Int}
    k_smoothmax::Float64
    epsilon::Float64
    N_TnTx::Int
end
smoothmax(x, k, maxkx) = (maxkx + log(sum(xi ->exp(k*xi - maxkx), x)))/k
smoothmin(x, k, minkx) = (minkx - log(sum(xi ->exp(-k*xi + minkx), x)))/k
smoothmax(x, k) = smoothmax(x, k, k*maximum(x))
smoothmin(x, k) = smoothmin(x, k, k*minimum(x))

dsmoothmax_dx(x, k, maxkx) = exp.( k.*x .- maxkx) ./ sum(xi ->exp( k*xi - maxkx), x)
dsmoothmin_dx(x, k, maxkx) = exp.(-k.*x .+ maxkx) ./ sum(xi ->exp(-k*xi + maxkx), x)
dsmoothmax_dx(x, k) = dsmoothmax_dx(x, k, k*maximum(x))
dsmoothmin_dx(x, k) = dsmoothmin_dx(x, k, k*minimum(x))
function loglik(pt::PredictTemperatures, θ::Vector)
    w_uncorr = θ
    temp_impt = pt.μ + pt.chol*w_uncorr
    
    dayend = cumsum(pt.impt_times_p_day)
    daystart = [1; dayend[1:end-1].-1]
    k = pt.k_smoothmax
    Tsmoothmin = [smoothmin(@view(temp_impt[daystart[i]:dayend[i]]), k)
                  for i in 1:pt.N_TnTx]
    Tsmoothmax = [smoothmax(@view(temp_impt[daystart[i]:dayend[i]]), k)
                          for i in 1:pt.N_TnTx]
    Tn_loglik = -sum(i -> (Tsmoothmin[i] - pt.Tn[i])^2, 1:pt.N_TnTx)  / (2*pt.epsilon^2)
    Tx_loglik = -sum(i -> (Tsmoothmax[i] - pt.Tx[i])^2 , 1:pt.N_TnTx) / (2*pt.epsilon^2)
    return Tn_loglik + Tx_loglik # + constant
end    
function loglik(pt::PredictTemperatures, θ)
    w_uncorr = θ
    temp_impt = pt.μ + pt.chol*w_uncorr
    
    dayend = cumsum(pt.impt_times_p_day)
    daystart = [1; dayend[1:end-1].-1]
    k = pt.k_smoothmax
    Tsmoothmin = [smoothmin(temp_impt[daystart[i]:dayend[i]], k)
                  for i in 1:pt.N_TnTx]
    Tsmoothmax = [smoothmax(temp_impt[daystart[i]:dayend[i]], k)
                          for i in 1:pt.N_TnTx]
    Tn_loglik = -sum(i -> (Tsmoothmin[i] - pt.Tn[i])^2, 1:pt.N_TnTx)  / (2*pt.epsilon^2)
    Tx_loglik = -sum(i -> (Tsmoothmax[i] - pt.Tx[i])^2 , 1:pt.N_TnTx) / (2*pt.epsilon^2)
    return Tn_loglik + Tx_loglik # + constant
end
function logprior(θ)
    return -sum(x -> x^2, θ)/2
end
function dlogpriordθ(θ)
    return -θ
end
logtarget(pt::PredictTemperatures, θ) = logprior(θ) + loglik(pt, θ)
logtarget_mat(pt::PredictTemperatures, θ) = logprior(θ) + loglik_mat(pt, θ)

function dloglikdθ(pt::PredictTemperatures, θ)
    temp_impt = pt.μ + pt.chol*θ
    
    dayend = cumsum(pt.impt_times_p_day)
    daystart = [1; dayend[1:end-1].+1]
    k = pt.k_smoothmax
    
    loglik = 0.0
    ntemp = length(temp_impt)
    dloglik_dtemp = Vector{Float64}(undef, ntemp)
    @inbounds for day in 1:pt.N_TnTx
        times = daystart[day]:dayend[day]
        temp_day = temp_impt[times]
        Tsmoothmin = smoothmin(temp_day, k)
        Tsmoothmax = smoothmax(temp_day, k)
        loglik +=  -(Tsmoothmin - pt.Tn[day])^2
        loglik +=  -(Tsmoothmax - pt.Tx[day])^2
        
        dloglik_dTsmoothmin = -(Tsmoothmin .- pt.Tn[day])
        dloglik_dTsmoothmax = -(Tsmoothmax .- pt.Tx[day])
        dloglik_dtemp[times]   = dloglik_dTsmoothmin .* dsmoothmin_dx(temp_day, k)
        dloglik_dtemp[times] .+= dloglik_dTsmoothmax .* dsmoothmax_dx(temp_day, k)
    end
    loglik /= (2*pt.epsilon^2)
    dloglik_dθ = dloglik_dtemp
    lmul!(pt.chol', dloglik_dθ)
    dloglik_dθ ./= (pt.epsilon^2)
    return loglik, vec(dloglik_dθ)
end

function dlogtargetdθ(pt::PredictTemperatures, θ)
    loglik, dloglik = dloglikdθ(pt, θ)
    dprior = dlogpriordθ(θ)
    prior = logprior(θ)
    return loglik+prior, dloglik.+dprior
end
# LogDensityProblem interface
dimension(pt::PredictTemperatures) = pt.Nimpt
function logdensity(::Type{LogDensityProblems.ValueGradient}, pt::PredictTemperatures, θ::AbstractVector)
    logtarget, dlogtarget = dlogtargetdθ(pt, θ)
    if !isfinite(logtarget)
        logtarget = -Inf
    else
        if !all(isfinite, dlogtarget)
            @show θ
        end
        @assert all(isfinite, dlogtarget)
    end
    return ValueGradient(logtarget, dlogtarget)
end
function logdensity(::Type{LogDensityProblems.Value}, pt::PredictTemperatures, θ::AbstractVector)
    val = Value(logtarget(pt, θ))
    if !isfinite(val)
        return -Inf
    end
end
