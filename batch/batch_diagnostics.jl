

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

function get_nearby(fw::FittingWindow, GPmodel::AbstractString, usaf::Int, wban::Int, icao::String, saved_dir::String; crossval::Bool)
    pred_dir = joinpath(saved_dir, "predictions_from_nearby", crossval ? "crossval" : "mll", GPmodel, icao)
    pred_fname = predictions_fname(usaf, wban, icao, fw)
    pred_fpath = joinpath(pred_dir, pred_fname)
    nearby_pred = load(pred_fpath)["nearby_pred"]
    return nearby_pred
end

function print_diagnostics(nearby::TemperatureImputations.NearbyPrediction,
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
                    Date(fw.start_date), Date(fw.end_date))
                    # Date(fw.start_date)-Day(0), Date(fw.end_date)-Day(0))
end

function t_inside_wt(t::DateTime, wt::WindowTime)
    return wt.start_time <= t <= wt.end_time
end
function t_inside_fw(t::DateTime, fw::FittingWindow, hr_measure::Hour)
    measure_day = TemperatureImputations.measurement_date(t, hr_measure)
    in_window = fw.start_date <= measure_day <= fw.end_date-Day(1)
    return in_window
end

function arg_test_fw(test::DataFrame, fw::FittingWindow, hr_measure::Hour)
    ts = test[:ts]
    in_window = t_inside_fw.(ts, Ref(fw), hr_measure)
    return in_window
end
function get_test_fw(test::DataFrame, fw::FittingWindow, hr_measure::Hour)
    in_window = arg_test_fw(test, fw, hr_measure)
    return test[in_window,:]
end
function get_window(windownum::Int, stan_increment::Day, stan_days::Day, hr_measure::Hour)
    janfirst = DateTime(2015, 1, 1, 0, 0, 0)
    mintime = DateTime(2015,1,1,0,0,0)
    maxtime = DateTime(2016,1,1,0,0,0)
    stan_start = janfirst + hr_measure - Day(1) + (windownum-1)*stan_increment
    stan_end = stan_start + stan_days
    stan_fw = FittingWindow(max(stan_start, mintime), min(stan_end, maxtime))
    return stan_fw
end
# function window_center(fw::FittingWindow, increment::Day)
    # # this doesn't generalize
    # return FittingWindow(fw.start_date+increment,fw.end_date-increment)
# end

function Chains(samples::AbstractArray{Float64, 3}, names::AbstractVector{S}) where {S<:AbstractString}
    nsamples, ncol, nchains = size(samples)
    chains = AxisArray(samples, Axis{:sample}(1:nsamples), Axis{:param}(names), Axis{:chain}(1:nchains))
    return chains
end

function read_ts(stan_fw_dir::String)
    ts_path = joinpath(stan_fw_dir, "timestamps.csv")
    ts = CSV.read(ts_path;  header=[:ts], datarow=1, allowmissing=:none)[:ts]
    ts::Vector{DateTime}
    return ts
end
function read_ts(fw::FittingWindow, GPmodel::AbstractString, usaf::Int, wban::Int, icao::String, saved_dir::String)
    stan_model_dir = joinpath(saved_dir,  "stan_fit", GPmodel)
    stan_fw_dir = joinpath(stan_model_dir, stan_dirname(usaf, wban, icao, fw))
    return read_ts(stan_fw_dir)
end
function read_ts(fw::FittingWindow, GPmodel::AbstractString, hr_measure_fals::Hour, usaf::Int, wban::Int, icao::String, saved_dir::String)
    stan_fw_dir = stan_dirname(usaf, wban, icao, fw)
    stan_model_dir = joinpath(SAVED_DIR, "hr_measure", GPmodel, str_hour(hr_measure_fals))
    return read_ts(joinpath(stan_model_dir, stan_fw_dir))
end
function get_chains_and_ts(stan_fw_dir::String)
    chains = let
        imputation_model = get_imputation_model(;pdir=stan_fw_dir, seed=-1)
        StanSample.read_samples(imputation_model; output_format=:mcmcchains)
    end
    ts = read_ts(stan_fw_dir)
    return chains, ts
end
function get_chains_and_ts(fw::FittingWindow, GPmodel::AbstractString, usaf::Int, wban::Int, icao::String, saved_dir::String; crossval::Bool)
    stan_model_dir = joinpath(saved_dir,  "stan_fit", crossval ? "crossval" : "mll", GPmodel)
    stan_fw_dir = joinpath(stan_model_dir, stan_dirname(usaf, wban, icao, fw))
    return get_chains_and_ts(stan_fw_dir)
end

str_hour(hr::Hour) = string(hr.value)
function get_chains_and_ts(fw::FittingWindow, GPmodel::AbstractString, hr_measure_fals::Hour, usaf::Int, wban::Int, icao::String, saved_dir::String; verbose=false)
    stan_model_dir = joinpath(SAVED_DIR, "hr_measure", GPmodel, str_hour(hr_measure_fals))
    stan_fw_dir = joinpath(stan_model_dir, stan_dirname(usaf, wban, icao, fw))
    return get_chains_and_ts(usaf, wban, icao, stan_fw_dir)
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
        nearby::TemperatureImputations.NearbyPrediction, 
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
