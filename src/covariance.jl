using Dates: Day
using GaussianProcesses: Folds, logp_CVfold, dlogpdθ_CVfold

function make_chunks_and_folds(k_spatiotemporal::Kernel, logNoise::Float64, 
                      stations_data::DataFrame, hourly_data::DataFrame;
                      window::Day)
    chunks=GPE[]
    chunk_width=24*(window/Day(1))
    tstart=0.0
    nobsv=0
    max_time = maximum(hourly_data.ts_hours)
    println("creating GP chunks")
    folds_reals = Folds[]
    while tstart < max_time
        tend=tstart+chunk_width
        in_chunk= tstart .<= hourly_data.ts_hours .< tend
        hourly_chunk = hourly_data[in_chunk,:]
        nobsv_chunk = sum(in_chunk)
        nobsv += nobsv_chunk

        chunk_X_PRJ = stations_data.X_PRJ[hourly_chunk.station]
        chunk_Y_PRJ = stations_data.Y_PRJ[hourly_chunk.station]
        chunk_X = [hourly_chunk.ts_hours chunk_X_PRJ chunk_Y_PRJ]

        y = hourly_chunk.temp
        chunk = GPE(chunk_X', y, MeanConst(mean(y)), k_spatiotemporal, logNoise)
        push!(chunks, chunk)
        
        station = hourly_chunk.station
        chunk_folds = [findall(isequal(statuniq), station) 
                for statuniq in unique(station)]
        push!(folds_reals, chunk_folds)

        tstart=tend
    end
    reals = TemperatureImputations.GPRealisations(chunks)
    return reals, folds_reals
end

function optim_kernel(k_spatiotemporal::Kernel, logNoise_init::Float64, 
                      stations_data::DataFrame, hourly_data::DataFrame, 
                      method::Symbol=:NLopt; 
                      window::Day,
                      x_tol=1e-5, f_tol=1e-10, kwargs...)
    reals, folds_reals = make_chunks_and_folds(k_spatiotemporal, logNoise_init, 
            stations_data, hourly_data; window=window)
    local min_neg_ll
    local min_hyp
    local opt_out
    println("begin optimization")
    if method == :NLopt
        min_neg_ll, min_hyp, ret, count = TemperatureImputations.optimize_NLopt(reals, domean=false, kern=true, noise=true, x_tol=x_tol, f_tol=f_tol)
        opt_out = (min_neg_ll, min_hyp, ret, count)
        converged = ret ∈ (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED)
    elseif method == :Optim
        opt_out = TemperatureImputations.optimize!(reals; domean=false, kern=true, noise=true,
                                      options=Optim.Options(;x_tol=x_tol, f_tol=f_tol, kwargs...)
                                     )
        min_hyp = Optim.minimizer(opt_out)
        min_neg_ll = Optim.minimum(opt_out)
        converged = Optim.converged(opt_out)
    else
        throw(MethodError())
    end
    @assert min_neg_ll ≈ -reals.mll
    return Dict(
        :hyp => min_hyp,
        :logNoise => reals.logNoise,
        :minimum => min_neg_ll,
        :opt_out => opt_out,
        :converged => converged,
       )
end

function optim_kernel_CV(k_spatiotemporal::Kernel, logNoise_init::Float64, 
                      stations_data::DataFrame, hourly_data::DataFrame, 
                      method::Symbol=:NLopt; 
                      window::Day,
                      x_tol=1e-5, f_tol=1e-10, kwargs...)
    reals, folds_reals = make_chunks_and_folds(k_spatiotemporal, logNoise_init, 
            stations_data, hourly_data; window=window)
    local min_neg_ll
    local min_hyp
    local opt_out
    println("begin optimization")
    if method == :NLopt
        min_neg_ll, min_hyp, ret, count = TemperatureImputations.optimize_NLopt_CV(
                reals, folds_reals,
                domean=false, kern=true, noise=true,
                x_tol=x_tol, f_tol=f_tol)
        opt_out = (min_neg_ll, min_hyp, ret, count)
        converged = ret ∈ (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED)
    elseif method == :Optim
        opt_out = TemperatureImputations.optimize_CV!(reals, folds_reals;
                domean=false, kern=true, noise=true,
                options=Optim.Options(;x_tol=x_tol, f_tol=f_tol, kwargs...)
                                     )
        min_hyp = Optim.minimizer(opt_out)
        min_neg_ll = Optim.minimum(opt_out)
        converged = Optim.converged(opt_out)
    else
        throw(MethodError())
    end
    return Dict(
        :hyp => min_hyp,
        :logNoise => reals.logNoise,
        :minimum => min_neg_ll,
        :opt_out => opt_out,
        :converged => converged,
       )
end
