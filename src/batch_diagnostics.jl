import Base.+
using TimeSeries
using DataTables
using GaussianProcesses
using GaussianProcesses: Mean, Kernel, evaluate, metric, IsotropicData, VecF64
using GaussianProcesses: Stationary, KernelData, MatF64, predict
import GaussianProcesses: optimize!, get_optim_target, cov, grad_slice!
import GaussianProcesses: num_params, set_params!, get_params, update_mll!, update_mll_and_dmll!
import GaussianProcesses: get_param_names, cov!, addcov!, multcov!
import Proj4
import Mamba
using JLD
using Distances
using DataTables: by, head
using Base.Dates: tonext, Hour, Day

function subset(df, from, to; closed_start=true, closed_end=true)
    ts = df[:ts].values
    return df[argsubset(ts,from,to;closed_start=closed_start,closed_end=closed_end),:]
end
function argsubset(ts, from, to; closed_start=true, closed_end=true)
    if closed_start
        after_from = ts .>= from
    else
        after_from = ts .> from
    end
    if closed_end
        before_to = ts .<= to
    else
        before_to = ts .< to
    end
    return after_from .& before_to
end

function add_diag!(Σ::PDMats.PDMat, a::Float64)
    mat = Σ.mat
    for i in 1:size(mat,1)
        mat[i,i] += a
    end
    copy!(Σ.chol.factors, mat)
    cholfact!(Σ.chol.factors, Symbol(Σ.chol.uplo))
    @assert sumabs(mat .- full(Σ.chol)) < 1e-8
    return Σ
end

type FittingWindow
    start_date::Date
    end_date::Date
end

function predictions_fname(usaf::Int, fw::FittingWindow)
    return @sprintf("%d_%s_to_%s.jld", 
                    usaf, fw.start_date, fw.end_date)
end

function get_nearby(fw::FittingWindow, GPmodel::AbstractString)
    saved_dir = joinpath(SAVED_DIR, "predictions_from_nearby", GPmodel)
    pred_fname = predictions_fname(test_usaf, fw)
    nearby_pred=load(joinpath(saved_dir, pred_fname))["nearby_pred"]
end

function print_diagnostics(nearby::TempModel.NearbyPrediction; ndraws=10000)
    ts = nearby.ts
    ts_start = minimum(ts)
    ts_end = maximum(ts)
    test_subset = subset(test_trimmed, ts_start, ts_end)
    train_subset = subset(hourly_train, ts_start, ts_end)

    μ = nearby.μ
    Σ = nearby.Σ
    nobsv = length(μ)
    
    centering = eye(nobsv) .- (1.0/nobsv) .* ones(nobsv, nobsv)
    Σ_centered = centering * Σ.mat * centering
    distr = MultivariateNormal(μ, Σ)
#     println("sum(Σ_centered)=", sum(Σ_centered*Σ_centered))
    for _ in 1:5
        temp_sim = rand(distr)
        println("var(predicted mean - simulated prediction)=", var(μ .- temp_sim))
    end
    temp_true = test_subset[:temp].values
    println("E(var(predicted mean - predictive draw))=", mean(var(μ .- rand(distr)) for _ in 1:ndraws))
    println("var(truth - predicted mean)=", var(μ .- temp_true))
end

function stan_dirname(usaf::Int, fw::FittingWindow)
    return @sprintf("%d_%s_to_%s/", 
                    usaf, fw.start_date, fw.end_date)
end

function get_test_fw(test::DataTable, fw::FittingWindow)
    ts = test[:ts].values
    in_window = [(fw.start_date <= TempModel.measurement_date(t, hr_measure) <= fw.end_date-Day(1)) for t in ts]
    return test[in_window,:]
end
function arg_test_fw(test::DataTable, fw::FittingWindow)
    ts = test[:ts].values
    in_window = [(fw.start_date <= TempModel.measurement_date(t, hr_measure) <= fw.end_date-Day(1)) for t in ts]
    return in_window
end
function get_window(windownum::Int)
    stan_start = janfirst + (windownum-1)*stan_increment
    stan_end = stan_start + stan_days
    stan_fw = FittingWindow(stan_start, stan_end)
    return stan_fw
end
function window_center(fw::FittingWindow, increment::Day)
    # this doesn't generalize
    return FittingWindow(fw.start_date+increment,fw.end_date-increment)
end

function get_chains_and_ts(fw::FittingWindow, GPmodel::AbstractString)
    stan_fw_dir = stan_dirname(test_usaf, fw)
    stan_model_dir = joinpath(SAVED_DIR, "stan_fit", GPmodel)
    stan_window_files = readdir(joinpath(stan_model_dir, stan_fw_dir))
    samplefiles = [joinpath(stan_model_dir, stan_fw_dir, f) for 
        f in stan_window_files if 
        (startswith(f,"imputation_samples") &
        endswith(f,".csv"))]
    header=String[]
    all_samples=[]
    for sf in samplefiles
        samples, header = readcsv(sf, Float64; header=true)
        push!(all_samples, samples)
    end

    samples = cat(3, all_samples...)
    chains = Mamba.Chains(samples; start=1, thin=1, chains=[i for i in 1:4], names=vec(header))
    
    ts_string = readcsv(joinpath(stan_model_dir, stan_fw_dir, "timestamps.csv"), String; header=false)
    ts = [DateTime(s, "yyyy-mm-ddTHH:MM:SS") for s in vec(ts_string)]
    return chains, ts
end

# convenience function to extract the imputed temperatures
# from the Mamba Chains object
function get_temperatures_reparam(chains::Mamba.Chains)
    temp_varnames = [h for h in chains.names if startswith(h, "temp_impt.")]
    temp_samples=getindex(chains, :, temp_varnames, :).value
    return temp_samples
end

function get_temp_percentiles(temp_impute)
    stacked_impute=vcat((temp_impute[:,:,i] for i in 1:size(temp_impute,3))...)
    sorted_impute = sort(stacked_impute,1)
    nsamples=size(sorted_impute,1)
    # extract  10th and 90th percentiles
    # of the imputations
    imputed_10 = sorted_impute[div(nsamples,10), :]
    imputed_90 = sorted_impute[nsamples-div(nsamples,10), :]
    return imputed_10, imputed_90
end
function get_temp_mean(temp_impute)
    stacked_impute=vcat((temp_impute[:,:,i] for i in 1:size(temp_impute,3))...)
    μ = vec(mean(stacked_impute, 1))
    return μ
end


mse(yhat::Vector,y::Vector) = mean((y.-yhat).^2)
verr(yhat::Vector,y::Vector) = var(y.-yhat)
type ImputationDiagnostic
    sumEVarError::Float64
    sumSE::Float64
    n::Int
end
EVarError(diag::ImputationDiagnostic) = diag.sumEVarError / diag.n
mse(diag::ImputationDiagnostic) = diag.sumSE / diag.n
function ImputationDiagnostic(temp_impute::Array{Float64,3}, test_truth::DataTable)
    stacked_impute=vcat((temp_impute[:,:,i] for i in 1:size(temp_impute,3))...)
    temp_true = test_truth[:temp].values
    ts = test_subsubset[:ts].values
    μ = vec(mean(stacked_impute, 1))
    evar_err = mean(mse(μ, stacked_impute[i,:]) for i in 1:size(stacked_impute,1))
    MSE = mse(μ, temp_true)
    n = length(μ)
    return ImputationDiagnostic(evar_err*n, MSE*n, n)
end

function +(diag1::ImputationDiagnostic,diag2::ImputationDiagnostic)
    return ImputationDiagnostic(
        diag1.sumEVarError + diag2.sumEVarError,
        diag1.sumSE + diag2.sumSE,
        diag1.n + diag2.n
        )
end

""" 
    Is window A inside of window B?
"""
function a_isinside_b(t::DateTime, b::FittingWindow)
    start_after = t >= b.start_date
    end_before = t <= b.end_date+Day(1)
    return start_after & end_before
end
"""
    How much buffer time is there on either side of the window?
"""
function buffer(t::DateTime, b::FittingWindow)
    start_diff = abs(t - b.start_date)
    end_diff = abs(t - (b.end_date+Day(1)))
    return min(start_diff, end_diff)
end
""" 
    Amongst a list of candidate windows `cand`, find the window that includes `wind`
    with the largest buffer on either sides.
"""
function find_best_window(t::DateTime, cands::Vector{FittingWindow})
    incl_wdows = [fw for fw in cands if a_isinside_b(t, fw)]
    buffers = [buffer(t, fw) for fw in incl_wdows]
    imax = indmax(buffers)
    best_window = incl_wdows[imax]
    return best_window
end

function get_diagnostics(nearby::TempModel.NearbyPrediction, ts_start::DateTime, ts_end::DateTime; 
                            ndraws=10000)

    test_subset = subset(test_trimmed, ts_start, ts_end, closed_start=true, closed_end=true)
    
    ts = nearby.ts
    iinside = argsubset(ts, ts_start, ts_end)
    ts = ts[iinside]
    μ = nearby.μ[iinside]
    Σ = full(nearby.Σ)[iinside,iinside]
    nobsv = length(μ)
    
    centering = eye(nobsv) .- (1.0/nobsv) .* ones(nobsv, nobsv)
    Σ_centered = centering * Σ * centering
    distr = MultivariateNormal(μ, Σ)
    temp_true = test_subset[:temp].values
    n = length(ts)
    return NearbyPredDiagnostic(
        mean(var(μ .- rand(distr)) for _ in 1:ndraws)*n,
        var(μ .- temp_true)*n,
        n
    )
end
