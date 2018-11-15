import Base.+
using TimeSeries
using DataFrames
using CSV
using GaussianProcesses
#=using GaussianProcesses: Mean, Kernel, evaluate, metric, IsotropicData=#
#=using GaussianProcesses: Stationary, KernelData, predict=#
#=import GaussianProcesses: num_params, set_params!, get_params, update_mll!, update_mll_and_dmll!=#
#=import GaussianProcesses: get_param_names=#
#=import Proj4=#
import PDMats
# import Mamba
using JLD
#=using Distances=#
using AxisArrays
using DataFrames: by, head
using Dates: tonext, Hour, Day
using LinearAlgebra: cholesky!, Hermitian
using LinearAlgebra
using Random
using Printf

const janfirst = Date(2015, 1, 1)
const stan_increment = Day(3)
const stan_days = Day(9)

function subset(df, from, to; closed_start=true, closed_end=true)
    ts = df[:ts]
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
    copyto!(Σ.chol.factors, mat)
    # cholfact!(Σ.chol.factors, Symbol(Σ.chol.uplo))
    cholesky!(Hermitian(Σ.chol.factors, Symbol(Σ.chol.uplo)))
    @assert sumabs(mat .- Matrix(Σ.chol)) < 1e-8
    return Σ
end

struct FittingWindow
    start_date::Date
    end_date::Date
end
struct WindowTime
    start_time::DateTime
    end_time::DateTime
end

#=function predictions_fname(usaf::Int, fw::FittingWindow)=#
    #=return @sprintf("%d_%s_to_%s.jld", =#
                    #=usaf, fw.start_date, fw.end_date)=#
#=end=#
function predictions_fname(usaf::Int, wban::Int, icao::String, fw::FittingWindow)
     @sprintf("%d_%d_%s_%s_to_%s.jld", 
        usaf, wban, icao,
        Date(fw.start_date), Date(fw.end_date))
end

function get_nearby(fw::FittingWindow, GPmodel::AbstractString, usaf::Int, wban::Int, icao::String, saved_dir::String)
    pred_dir = joinpath(saved_dir, "predictions_from_nearby", GPmodel, icao)
    pred_fname = predictions_fname(usaf, wban, icao, fw)
    pred_fpath = joinpath(pred_dir, pred_fname)
    nearby_pred = load(pred_fpath)["nearby_pred"]
    return nearby_pred
end

function print_diagnostics(nearby::TempModel.NearbyPrediction,
        test_data, train_data; ndraws=10000)
    ts = nearby.ts
    ts_start = minimum(ts)
    ts_end = maximum(ts)
    test_subset = subset(test_data, ts_start, ts_end)
    train_subset = subset(train_data, ts_start, ts_end)

    μ = nearby.μ
    Σ = nearby.Σ
    nobsv = length(μ)
    
    centering = Matrix(1.0I, nobsv, nobsv) .- (1.0/nobsv)
    Σ_centered = centering * Σ.mat * centering
    distr = MultivariateNormal(μ, Σ)
#     println("sum(Σ_centered)=", sum(Σ_centered*Σ_centered))
    for _ in 1:5
        temp_sim = rand(distr)
        @printf("var(predicted mean - simulated prediction)= %.3f\n", var(μ .- temp_sim))
    end
    temp_true = test_subset[:temp]
    @printf("E(var(predicted mean - predictive draw))= %.3f\n", mean(var(μ .- rand(distr)) for _ in 1:ndraws))
    @printf("var(truth - predicted mean)= %.3f\n", var(μ .- temp_true))
end

function stan_dirname(usaf::Int, wban::Int, icao::String, fw::FittingWindow)
    return @sprintf("%s/%d_%d_%s_%s_to_%s/", 
                    icao, usaf, wban, icao,
                    Date(fw.start_date)-Day(1), Date(fw.end_date)-Day(1))
                    # Date(fw.start_date)-Day(0), Date(fw.end_date)-Day(0))
end

function t_inside_wt(t::DateTime, wt::WindowTime)
    return wt.start_time <= t <= wt.end_time
end
function t_inside_fw(t::DateTime, fw::FittingWindow, hr_measure::Hour)
    measure_day = TempModel.measurement_date(t, hr_measure)
    in_window = fw.start_date <= measure_day <= fw.end_date-Day(1)
    return in_window
end

function arg_test_fw(test::DataFrame, fw::FittingWindow, hr_measure::Hour)
    ts = test[:ts]
    in_window = t_inside_fw.(ts, fw, hr_measure)
    return in_window
end
function get_test_fw(test::DataFrame, fw::FittingWindow, hr_measure::Hour)
    in_window = arg_test_fw(test, fw, hr_measure)
    return test[in_window,:]
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

function Chains(samples::AbstractArray{Float64, 3}, names::AbstractVector{S}) where {S<:AbstractString}
    nsamples, ncol, nchains = size(samples)
    chains = AxisArray(samples, Axis{:sample}(1:nsamples), Axis{:param}(names), Axis{:chain}(1:nchains))
    return chains
end

function read_ts(fw::FittingWindow, GPmodel::AbstractString, usaf::Int, wban::Int, icao::String, saved_dir::String)
    stan_fw_dir = stan_dirname(usaf, wban, icao, fw)
    stan_model_dir = joinpath(saved_dir,  "stan_fit", GPmodel)
    stan_window_files = readdir(joinpath(stan_model_dir, stan_fw_dir))
    ts_path = joinpath(stan_model_dir, stan_fw_dir, "timestamps.csv")
    ts = CSV.read(ts_path, DataFrame;  header=[:ts], datarow=1, allowmissing=:none)[:ts]
    ts::Vector{DateTime}
    return ts
end
function read_samples_csv(samples_path::String)
    types = Dict(:lp__ => Float64, :accept_stat__ => Float64,
                 :stepsize__ => Float64, :treedepth__ => Int64,
                 :n_leapfrog__ => Int64, :divergent__ => Int64,
                 :energy__ => Float64)
    for i in 1:1000 # arbitrary number of times
        types[Symbol("w_uncorr.", i)] = Float64
        types[Symbol("temp_impt.", i)] = Float64
        types[Symbol("Tsmoothmin.", i)] = Float64
        types[Symbol("Tsmoothmax.", i)] = Float64
    end
    samples_df = CSV.File(samples_path, 
            comment="#",
            header=38, # hacky pending https://github.com/JuliaData/CSV.jl/issues/351
            skipto=43, # hacky
            allowmissing=:none, types=types,
            ) |> DataFrame;
    data = Matrix{Float64}(samples_df)
    header = String.(names(samples_df))
    # DelimitedFiles.readdlm is too slow
    # data, header = DelimitedFiles.readdlm(samples_path, ',', Float64, '\n'; comments=true, header=true) 
    return data, header
end
function get_chains_and_ts(fw::FittingWindow, GPmodel::AbstractString, usaf::Int, wban::Int, icao::String, saved_dir::String)
    stan_fw_dir = stan_dirname(usaf, wban, icao, fw)
    stan_model_dir = joinpath(saved_dir, "stan_fit", GPmodel)
    stan_window_files = readdir(joinpath(stan_model_dir, stan_fw_dir))
    samplefiles = [joinpath(stan_model_dir, stan_fw_dir, f) for 
        f in stan_window_files if 
        (startswith(f,"imputation_samples") &
        endswith(f,".csv"))]
    header=String[]
    all_samples=Matrix{Float64}[]
    for sf in samplefiles
        s, header = read_samples_csv(sf)
        push!(all_samples, s)
    end

    samples = cat(all_samples..., dims=3)
    # chains = Mamba.Chains(samples; start=1, thin=1, chains=[i for i in 1:4], names=vec(header))
    chains = Chains(samples, vec(header))
    
    ts = read_ts(fw, GPmodel, usaf, wban, icao, saved_dir)
    return chains, ts
end

str_hour(hr::Hour) = string(hr.value)
function read_ts(fw::FittingWindow, GPmodel::AbstractString, hr_measure_fals::Hour, usaf::Int, wban::Int, icao::String, saved_dir::String)
    stan_fw_dir = stan_dirname(usaf, wban, icao, fw)
    stan_model_dir = joinpath(SAVED_DIR, "hr_measure", GPmodel, str_hour(hr_measure_fals))
    stan_window_files = readdir(joinpath(stan_model_dir, stan_fw_dir))
    ts_path = joinpath(stan_model_dir, stan_fw_dir, "timestamps.csv")
    ts = CSV.read(ts_path, DataFrame;  header=[:ts], datarow=1, allowmissing=:none)[:ts]
    ts::Vector{DateTime}
    return ts
end
function get_chains_and_ts(fw::FittingWindow, GPmodel::AbstractString, hr_measure_fals::Hour, usaf::Int, wban::Int, icao::String, saved_dir::String; verbose=false)
    stan_fw_dir = stan_dirname(usaf, wban, icao, fw)
    stan_model_dir = joinpath(SAVED_DIR, "hr_measure", GPmodel, str_hour(hr_measure_fals))
    if verbose
        @show joinpath(stan_model_dir, stan_fw_dir)
    end
    stan_window_files = readdir(joinpath(stan_model_dir, stan_fw_dir))
    if verbose
        @show stan_window_files
    end
    samplefiles = [joinpath(stan_model_dir, stan_fw_dir, f) for 
        f in stan_window_files if 
        (startswith(f,"imputation_samples") &
        endswith(f,".csv"))]
    header=String[]
    all_samples=Matrix{Float64}[]
    for sf in samplefiles
        s, header = read_samples_csv(sf)
        push!(all_samples, s)
    end
    @assert length(all_samples) > 1

    samples = cat(all_samples..., dims=3)
    chains = Chains(samples, vec(header))
    ts = read_ts(fw, GPmodel, hr_measure_fals, usaf, wban, icao, saved_dir)
    return chains, ts
end

# convenience function to extract the imputed temperatures
# from the Mamba Chains object
# function get_temperatures_reparam(chains::Mamba.Chains)
    # temp_varnames = [h for h in chains.names if startswith(h, "temp_impt.")]
    # temp_samples=getindex(chains, :, temp_varnames, :).value
    # return temp_samples
# end
function get_temperatures_reparam(chains::DataFrame)
    temp_varnames = [h for h in names(chains) if startswith(h, "temp_impt.")]
    temp_samples=getindex(chains, :, temp_varnames, :)
    return temp_samples
end
function get_param_names(chains::AxisArray)
    # there should be a more elegant way to obtain the names of an axis
    PARAM = Axis{:param}
    jparam = axisdim(chains, PARAM)
    param_axis = AxisArrays.axes(chains)[jparam]
    param_names = axisvalues(param_axis)[1]
    return param_names
end
function get_temperatures_reparam(chains::AxisArray)
    # next 4 lines: tedious way to get names of parameters
    param_names = get_param_names(chains)
    temp_varnames = [h for h in param_names if startswith(h, "temp_impt.")]
    temp_samples = view(chains, :, Axis{:param}(temp_varnames), :)
    return temp_samples
end

function get_temp_percentiles(temp_impute)
    stacked_impute=vcat((temp_impute[:,:,i] for i in 1:size(temp_impute,3))...)
    sorted_impute = sort(stacked_impute, dims=1)
    nsamples=size(sorted_impute,1)
    # extract  10th and 90th percentiles
    # of the imputations
    imputed_10 = sorted_impute[div(nsamples,10), :]
    imputed_90 = sorted_impute[nsamples-div(nsamples,10), :]
    return imputed_10, imputed_90
end
function get_temp_mean(temp_impute)
    stacked_impute=vcat((temp_impute[:,:,i] for i in 1:size(temp_impute,3))...)
    μ = vec(mean(stacked_impute, dims=1))
    return μ
end


mse(yhat::Vector,y::Vector) = mean((y.-yhat).^2)
verr(yhat::Vector,y::Vector) = var(y.-yhat)
struct ImputationDiagnostic
    sumEVarError::Float64
    sumSE::Float64
    n::Int
end
EVarError(diag::ImputationDiagnostic) = diag.sumEVarError / diag.n
mse(diag::ImputationDiagnostic) = diag.sumSE / diag.n
function ImputationDiagnostic(temp_impute::AbstractArray{Float64,3}, test_truth::DataFrame)
    stacked_impute=vcat((temp_impute[:,:,i] for i in 1:size(temp_impute,3))...)
    temp_true = test_truth[:temp]
    μ = vec(mean(stacked_impute, dims=1))
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
    How much buffer time is there on either side of the window?
"""
function buffer(t::DateTime, wt::WindowTime)
    start_diff = t - wt.start_time
    end_diff = wt.end_time - t
    return min(start_diff, end_diff)
end
""" 
    Amongst a list of candidate windows `cand`, find the window that includes `wind`
    with the largest buffer on either sides.
"""
function find_best_window(t::DateTime, cands::Vector{WindowTime})
    inside_windows = t_inside_wt.(t, cands)
    incl_wdows = cands[inside_windows]
    buffers = [buffer(t, wt) for wt in incl_wdows]
    imax = argmax(buffers)
    return find(inside_windows)[imax]
end

function daily_best(all_posteriors)
    unique_days = Set{Date}()
    for day_post in all_posteriors
        day_post["days"] = Date.(day_post["days"], "yyyy-mm-dd")
        day_post["cov"] = hcat(day_post["cov"]...)::Matrix{Float64}
        union!(unique_days, day_post["days"])
    end

    days_vec = sort(collect(unique_days))
    ndays = length(unique_days)
    day_buffer = fill(0, ndays)
    day_means = Vector{Float64}(undef, ndays)
    day_cov = zeros(Float64, ndays, ndays)
    buffer_cov = fill(0, ndays, ndays)

    for day_post in all_posteriors
        days = day_post["days"]
        post_mean, post_cov = day_post["mean"], day_post["cov"]
        buffer = min.(1:1:length(days), length(days):-1:1)
        cross_buffer = min.(buffer, buffer')
        ifirst = findfirst(isequal(days[1]), days_vec)
        days_indices = ifirst:(ifirst+length(days)-1)
        is_better = findall(buffer .> day_buffer[days_indices])
        better_indices = days_indices[is_better]
        
        cov_view_days = @view(day_cov[days_indices,days_indices])
        buf_cov_view  = @view(buffer_cov[days_indices,days_indices])
        is_better_cov = findall(cross_buffer .> buf_cov_view)
        
        day_means[better_indices] = post_mean[is_better]
        day_buffer[better_indices] = buffer[is_better]
        copyto!(@view(cov_view_days[is_better_cov]), post_cov[is_better_cov])
        copyto!(@view(buf_cov_view[is_better_cov]), cross_buffer[is_better_cov])
    end
    @assert minimum(day_buffer) > 0
    return days_vec, day_means, day_cov, day_buffer, buffer_cov
end

#######################################
###### Nearby-only predictions ########
#######################################

struct NearbyPredDiagnostic
    sumEVarError::Float64
    sumVarError::Float64
    n::Int
end
function +(diag1::NearbyPredDiagnostic,diag2::NearbyPredDiagnostic)
    return NearbyPredDiagnostic(
        diag1.sumEVarError + diag2.sumEVarError,
        diag1.sumVarError + diag2.sumVarError,
        diag1.n + diag2.n
        )
end
function get_diagnostics(
        nearby::TempModel.NearbyPrediction, 
        test_data, 
        ts_start::DateTime, ts_end::DateTime; 
        ndraws=10000)

    test_subset = subset(test_data, ts_start, ts_end, closed_start=true, closed_end=true)
    
    ts = nearby.ts
    iinside = argsubset(ts, ts_start, ts_end)
    ts = ts[iinside]
    μ = nearby.μ[iinside]
    Σ = PDMats.full(nearby.Σ)[iinside,iinside]
    nobsv = length(μ)
    
    centering = Matrix(1.0I, nobsv, nobsv) .- (1.0/nobsv)
    Σ_centered = centering * Σ * centering
    distr = MultivariateNormal(μ, Σ)
    temp_true = test_subset[:temp]
    n = length(ts)
    return NearbyPredDiagnostic(
        mean(var(μ .- rand(distr)) for _ in 1:ndraws)*n,
        var(μ .- temp_true)*n,
        n
    )
end
