using GaussianProcesses: Mean, Kernel, evaluate, metric
import GaussianProcesses: optimize!, get_optim_target
import GaussianProcesses: num_params, set_params!, get_params, update_mll!
import GaussianProcesses: update_mll_and_dmll!
using Optim: minimizer

type GPRealisations
    reals::Vector{GP}
    m::Mean
    k::Kernel
    logNoise::Float64
    mLL::Float64
    dmLL::Vector{Float64}
end

function GPRealisations(reals::Vector{GP})
    first = reals[1]
    gpr = GPRealisations(reals, first.m, first.k, first.logNoise, NaN, [])
end

function get_params(gpr::GPRealisations; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; push!(params, gpr.logNoise); end
    if mean;  append!(params, get_params(gpr.m)); end
    if kern; append!(params, get_params(gpr.k)); end
    return params
end
function propagate_params!(gpr::GPRealisations; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    for gp in gpr.reals
        # harmonize parameters
        if kern
            gp.k = gpr.k
        end
        if mean
            gp.m = gpr.m
        end
        if noise
            gp.logNoise = gpr.logNoise
        end
    end
end
function set_params!(gpr::GPRealisations, hyp::Vector{Float64}; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    # println("mean=$(mean)")
    if noise; gpr.logNoise = hyp[1]; end
    if mean; set_params!(gpr.m, hyp[1+noise:noise+num_params(gpr.m)]); end
    if kern; set_params!(gpr.k, hyp[end-num_params(gpr.k)+1:end]); end
    propagate_params!(gpr, noise=noise, mean=mean, kern=kern)
end

function update_mll!(gpr::GPRealisations)
    mLL = 0.0
    for gp in gpr.reals
        update_mll!(gp)
        mLL += gp.mLL
    end
    gpr.mLL = mLL
    return mLL
end
function update_mll_and_dmll!(gpr::GPRealisations, Kgrads::Dict{Int,Matrix}, ααinvcKIs::Dict{Int,Matrix}; kwargs...)
    mLL = 0.0
    for gp in gpr.reals
        update_mll_and_dmll!(gp, Kgrads[gp.nobsv], ααinvcKIs[gp.nobsv]; kwargs...)
        mLL += gp.mLL
    end
    gpr.dmLL = gpr.reals[1].dmLL
    for gp in gpr.reals[2:end]
        gpr.dmLL .+= gp.dmLL
    end
    gpr.mLL = mLL
    return gpr.dmLL
end
function update_mll_and_dmll!(gpr::GPRealisations; kwargs...)
    Kgrads = Dict{Int,Matrix}()
    ααinvcKIs = Dict{Int,Matrix}()
    for gp in gpr.reals
        if haskey(Kgrads, gp.nobsv)
            continue
        end
        Kgrads[gp.nobsv] = Array(Float64, gp.nobsv, gp.nobsv)
        ααinvcKIs[gp.nobsv] = Array(Float64, gp.nobsv, gp.nobsv)
    end
    return update_mll_and_dmll!(gpr, Kgrads, ααinvcKIs)
end

function get_optim_target(gpr::GPRealisations; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    Kgrads = Dict{Int,Matrix}()
    ααinvcKIs = Dict{Int,Matrix}()
    for gp in gpr.reals
        if haskey(Kgrads, gp.nobsv)
            continue
        end
        Kgrads[gp.nobsv] = Array(Float64, gp.nobsv, gp.nobsv)
        ααinvcKIs[gp.nobsv] = Array(Float64, gp.nobsv, gp.nobsv)
    end
    function mll(hyp::Vector{Float64})
        try
            set_params!(gpr, hyp; noise=noise, mean=mean, kern=kern)
            update_mll!(gpr)
            return -gpr.mLL
        catch err
             if !all(isfinite(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end        
    end

    function mll_and_dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        try
            set_params!(gpr, hyp; noise=noise, mean=mean, kern=kern)
            update_mll_and_dmll!(gpr, Kgrads, ααinvcKIs; noise=noise, mean=mean, kern=kern)
            grad[:] = -gpr.dmLL
            return -gpr.mLL
        catch err
             if !all(isfinite(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end 
    end
    function dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        mll_and_dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
    end

    func = DifferentiableFunction(mll, dmll!, mll_and_dmll!)
    return func
end
function optimize!(gpr::GPRealisations; noise::Bool=true, mean::Bool=true, kern::Bool=true,
                    method=ConjugateGradient(), kwargs...)
    func = get_optim_target(gpr, noise=noise, mean=mean, kern=kern)
    init = get_params(gpr;  noise=noise, mean=mean, kern=kern)  # Initial hyperparameter values
    results=optimize(func,init; method=method, kwargs...)                     # Run optimizer
    set_params!(gpr, minimizer(results), noise=noise,mean=mean,kern=kern)
    update_mll!(gpr)
    return results
end
