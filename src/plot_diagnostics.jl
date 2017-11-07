function plot_imputations(ts, temp_impute)
    imputed_10, imputed_90 = get_temp_percentiles(temp_impute)
    μ = get_temp_mean(temp_impute)
    
    plt.fill_between(ts, imputed_10, imputed_90, 
        color="#F16424", alpha=0.5, label=L"$80\%$ credible interval")
    plt.plot(ts, temp_impute[250,:,1],
        color="#009F77", label="single imputation")
    plt.plot(ts, μ, color="#F16424", linewidth=3, label="posterior mean")
end
function plot_truth(test::DataTable)
    ts = test[:ts].values
    plt.plot(ts, test[:temp].values, 
        color="black", "o-", label="true hourly")
    plt.step(ts, test_subsubset[:Tn].values, where="post",
        color="blue", linewidth=3, label=L"$T_n$")
    plt.step(ts, test_subsubset[:Tx].values, where="post",
        color="red", linewidth=3, label=L"$T_x$")
end
function plot_predictive(nearby_pred::TempModel.NearbyPrediction, xlim::Tuple{DateTime,DateTime};
                         truth::Bool=true,
                         neighbours::Bool=true,
                         mean_impt::Bool=true,
                         imputations::Bool=true)

    test_subset = subset(test_trimmed, xlim[1], xlim[2])
    train_subset = subset(hourly_train, xlim[1], xlim[2])
    μ = nearby_pred.μ
    Σ = nearby_pred.Σ
    nobsv = length(μ)
    
    centering = eye(nobsv) .- (1.0/nobsv) .* ones(nobsv, nobsv)
    Σ_centered = centering * Σ.mat * centering
    distr = MultivariateNormal(μ, Σ)
    if neighbours
        for station in unique(train_subset[:station].values)
            sdata = train_subset[train_subset[:station].values.==station,:]
            ts=sdata[:ts].values
            plt.plot(ts, sdata[:temp].values.-mean(sdata[:temp].values), color="#F8A21F", 
                label=get(isdSubset[station,:NAME]))
        end
    end
    ts=nearby_pred.ts
    in_window = (xlim[1] .<= ts) .& (ts .<= xlim[2])
    mean_μ = mean(μ[in_window])
    if mean_impt
        plt.plot(ts, μ-mean_μ, color="#009F77", linewidth=2, label="posterior mean")
        plt.fill_between(ts, 
                         μ-mean_μ-2*sqrt.(diag(Σ_centered)),
                         μ-mean_μ+2*sqrt.(diag(Σ_centered)),
                         color="#009F77", alpha=0.3)
    end
    if imputations
        for _ in 1:2
            temp_sim = rand(distr)
            mean_sim = mean(temp_sim[in_window])
            plt.plot(ts, temp_sim.-mean_sim, color="#009F77", linewidth=1)
        end
    end
    ts = test_subset[:ts].values
    temp_true = test_subset[:temp].values
    if truth
        mean_true = mean(temp_true)
        plt.plot(ts, temp_true.-mean_true, "o-",
            color="black", label="true hourly")
    end
    plt.legend(loc="best")
    plt.title("Predictive distribution based on nearby data alone")
    plt.gca()[:xaxis][:set_major_formatter](plt.matplotlib[:dates][:DateFormatter]("%Y-%m-%d"))
    plt.xticks(collect(plot_xlim[1]:Day(1):plot_xlim[2]))
    plt.xlim(xlim)
end
function plot_residuals(nearby::TempModel.NearbyPrediction)
    ts = nearby.ts
    ts_start = minimum(ts)
    ts_end = maximum(ts)
    test_subset = subset(test_trimmed, ts_start, ts_end)

    μ = nearby_pred.μ
    Σ = nearby_pred.Σ
    nobsv = length(μ)
    
    centering = eye(nobsv) .- (1.0/nobsv) .* ones(nobsv, nobsv)
    Σ_centered = centering * Σ.mat * centering
    distr = MultivariateNormal(μ, Σ)
    temp_true = test_subset[:temp].values
    println("var(truth - predicted mean)=", var(μ .- temp_true))
    residuals = μ .- temp_true
    plt.plot(ts, abs(residuals .- mean(residuals)))
    plt.ylabel("Absolute residuals")
end

