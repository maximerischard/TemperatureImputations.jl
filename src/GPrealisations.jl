using GaussianProcesses: Mean, Kernel, evaluate, metric
import GaussianProcesses: optimize!, get_optim_target
import GaussianProcesses: num_params, set_params!, get_params, update_mll!
import GaussianProcesses: update_mll_and_dmll!
using Optim: minimizer

type GPRealisations
    reals::Vector{GPE}
    m::Mean
    k::Kernel
    logNoise::Float64
    mll::Float64
    dmll::Vector{Float64}
end

function GPRealisations(reals::Vector{GPE})
    first = reals[1]
    gpr = GPRealisations(reals, first.m, first.k, first.logNoise, NaN, [])
end

function get_params(gpr::GPRealisations; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; push!(params, gpr.logNoise); end
    if domean;  append!(params, get_params(gpr.m)); end
    if kern; append!(params, get_params(gpr.k)); end
    return params
end
function propagate_params!(gpr::GPRealisations; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    for gp in gpr.reals
        # harmonize parameters
        if kern
            gp.k = gpr.k
        end
        if domean
            gp.m = gpr.m
        end
        if noise
            gp.logNoise = gpr.logNoise
        end
    end
end
function set_params!(gpr::GPRealisations, hyp::Vector{Float64}; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    # println("domean=$(domean)")
    if noise; gpr.logNoise = hyp[1]; end
    if domean; set_params!(gpr.m, hyp[1+noise:noise+num_params(gpr.m)]); end
    if kern; set_params!(gpr.k, hyp[end-num_params(gpr.k)+1:end]); end
    propagate_params!(gpr, noise=noise, domean=domean, kern=kern)
end

function update_mll!(gpr::GPRealisations)
    mll = 0.0
    for gp in gpr.reals
        update_mll!(gp)
        mll += gp.mll
    end
    gpr.mll = mll
    return mll
end
function update_mll_and_dmll!(gpr::GPRealisations, Kgrads::Dict{Int,Matrix}, ααinvcKIs::Dict{Int,Matrix}; kwargs...)
    mll = 0.0
    for gp in gpr.reals
        update_mll_and_dmll!(gp, Kgrads[gp.nobsv], ααinvcKIs[gp.nobsv]; kwargs...)
        mll += gp.mll
    end
    gpr.dmll = gpr.reals[1].dmll
    for gp in gpr.reals[2:end]
        gpr.dmll .+= gp.dmll
    end
    gpr.mll = mll
    return gpr.dmll
end
function update_mll_and_dmll!(gpr::GPRealisations; kwargs...)
    Kgrads = Dict{Int,Matrix}()
    ααinvcKIs = Dict{Int,Matrix}()
    for gp in gpr.reals
        if haskey(Kgrads, gp.nobsv)
            continue
        end
        Kgrads[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
        ααinvcKIs[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
    end
    return update_mll_and_dmll!(gpr, Kgrads, ααinvcKIs; kwargs...)
end

function get_optim_target(gpr::GPRealisations; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    Kgrads = Dict{Int,Matrix}()
    ααinvcKIs = Dict{Int,Matrix}()
    for gp in gpr.reals
        if haskey(Kgrads, gp.nobsv)
            continue
        end
        Kgrads[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
        ααinvcKIs[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
    end
    function mll(hyp::Vector{Float64})
        try
            set_params!(gpr, hyp; noise=noise, domean=domean, kern=kern)
            update_mll!(gpr)
            if !isfinite(gpr.mll)
                return Inf
            end
            return -gpr.mll
        catch err
             if !all(isfinite.(hyp))
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

    function mll_and_dmll!(grad::Vector{Float64}, hyp::Vector{Float64})
        try
            set_params!(gpr, hyp; noise=noise, domean=domean, kern=kern)
            update_mll_and_dmll!(gpr, Kgrads, ααinvcKIs; noise=noise, domean=domean, kern=kern)
            grad[:] = -gpr.dmll
            if !isfinite(gpr.mll)
                return Inf
            end
            return -gpr.mll
        catch err
             if !all(isfinite.(hyp))
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
    function dmll!(grad::Vector{Float64}, hyp::Vector{Float64})
        mll_and_dmll!(grad::Vector{Float64}, hyp::Vector{Float64})
    end

    func = OnceDifferentiable(mll, dmll!, mll_and_dmll!,
        get_params(gpr, noise=noise, domean=domean, kern=kern))
    return func
end
function optimize!(gpr::GPRealisations; noise::Bool=true, domean::Bool=true, kern::Bool=true,
                    method=ConjugateGradient(), options=Optim.Options())
    func = get_optim_target(gpr, noise=noise, domean=domean, kern=kern)
    init = get_params(gpr;  noise=noise, domean=domean, kern=kern)  # Initial hyperparameter values
    results=optimize(func,init,method, options)  # Run optimizer
    set_params!(gpr, minimizer(results), noise=noise,domean=domean,kern=kern)
    update_mll!(gpr)
    return results
end

function optimize_NLopt(gpr::GPRealisations; noise::Bool=true, domean::Bool=true, kern::Bool=true,
                    method=:LD_LBFGS, x_tol=1e-10, f_tol=1e-10)
    target = get_optim_target(gpr, noise=noise, domean=domean, kern=kern)
    init_x = get_params(gpr;  noise=noise, domean=domean, kern=kern)  # Initial hyperparameter values
    count = 0
    best_x = copy(init_x)
    best_y = Inf
    function myfunc(x::Vector, grad::Vector)
        if length(grad) > 0
            target.df(grad, x)
        end

        count += 1
        y = target.f(x)
        if isfinite(y) & (y < best_y)
            best_y = y
            best_x[:] = x
        end
        return y
    end

    nparams = length(init_x)
    opt = NLopt.Opt(method, nparams)

    lower = Array{Float64}(nparams)
    upper = Array{Float64}(nparams)
    lower = init_x - 3.0
    upper = init_x + 3.0
    NLopt.lower_bounds!(opt, lower)
    NLopt.upper_bounds!(opt, upper)
    NLopt.xtol_rel!(opt, x_tol)
    NLopt.ftol_rel!(opt, f_tol)
    NLopt.min_objective!(opt, myfunc)
    (minf,minx,ret) = NLopt.optimize(opt, init_x)

    set_params!(gpr, minx, noise=noise,domean=domean,kern=kern)
    update_mll!(gpr)
    return minf,minx,ret,count
end
