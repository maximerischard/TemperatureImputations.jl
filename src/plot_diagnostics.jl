cbbPalette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
colour_hourly_nearby = cbbPalette[1]
colour_truth = "black"
colour_pred_nearby = cbbPalette[7]
colour_pred_tntx = cbbPalette[3]
colour_impt_nearby = cbbPalette[7]
colour_impt_tntx = cbbPalette[3]
colour_tn = cbbPalette[5]
colour_tx = "red"

function plot_imputations(ts, temp_impute; impt_indices=[250])
    imputed_10, imputed_90 = get_temp_percentiles(temp_impute)
    μ = get_temp_mean(temp_impute)
    
    plt.fill_between(ts, imputed_10, imputed_90, 
        color=colour_pred_tntx, alpha=0.3)
    for i in impt_indices
        plt.plot(ts, temp_impute[i,:,1],
            color=colour_impt_tntx)
    end
    plt.plot(ts, μ, color=colour_pred_tntx, linewidth=3, 
             label=L"$\mathrm{T}_\mathrm{miss} \mid \mathrm{T}_\mathrm{nearby}, \mathrm{T}_{n}, \mathrm{T}_{x}$")
end
function plot_truth(test::DataFrame, window::FittingWindow; tntx::Bool=false, markersize=5)
    test_window = get_test_fw(test, window)
    ts = test_window[:ts].values
    temp = test_window[:temp].values
    Tn = test_window[:Tn].values
    Tx = test_window[:Tx].values
    plt.plot(ts, temp,
        marker="o",
        markersize=markersize,
        color=colour_truth, label="true hourly")
    if tntx
        plt.step(ts, Tn, where="pre",
            color=colour_tn, linewidth=3, label=L"$T_n$",
            zorder=-1)
        plt.step(ts, Tx, where="pre",
            color=colour_tx, linewidth=3, label=L"$T_x$",
            zorder=-1)
    end
end
function plot_predictive(nearby_pred::TempModel.NearbyPrediction, xlim::Tuple{DateTime,DateTime};
                         truth::Bool=true,
                         neighbours::Bool=true,
                         mean_impt::Bool=true,
                         imputations::Int=0,
                         markersize=5
                         )

    test_subset = subset(test_trimmed, xlim[1], xlim[2])
    train_subset = subset(hourly_train, xlim[1], xlim[2])
    μ = nearby_pred.μ
    Σ = nearby_pred.Σ
    nobsv = length(μ)
    
    centering = eye(nobsv) .- (1.0/nobsv) .* ones(nobsv, nobsv)
    Σ_centered = centering * Σ.mat * centering
    distr = MultivariateNormal(μ, Σ)
    if neighbours
        markers = ["v", "1", "p", "s"]
        for (i,station) in enumerate(unique(train_subset[:station].values))
            sdata = train_subset[train_subset[:station].values.==station,:]
            ts=sdata[:ts].values
            plt.plot(ts, sdata[:temp].values.-mean(sdata[:temp].values), 
                     color=colour_hourly_nearby, 
                     marker = markers[i],
                     markersize=markersize,
                     label=get(isdSubset[station,:NAME]),
                     zorder = 10
                     )
        end
    end
    ts=nearby_pred.ts
    in_window = (xlim[1] .<= ts) .& (ts .<= xlim[2])
    mean_μ = mean(μ[in_window])
    for i in 1:imputations
        temp_sim = rand(distr)
        mean_sim = mean(temp_sim[in_window])
        label = ""
        # if i==1
            # label = "example imputation"
        # end
        plt.plot(ts, temp_sim.-mean_sim, color=colour_impt_nearby, linewidth=1,
                 label=label,
                 zorder=20)
    end
    if mean_impt
        plt.plot(ts, μ-mean_μ, color=colour_pred_nearby, linewidth=2, 
                 zorder=30,
                 label=L"$\mathrm{T}_\mathrm{miss} \mid \mathrm{T}_\mathrm{nearby}$")

        intvl_width = 0.8
        intvl_stds = -quantile(Normal(0,1), (1-intvl_width)/2)

        plt.fill_between(ts, 
                         μ-mean_μ-intvl_stds*sqrt.(diag(Σ_centered)),
                         μ-mean_μ+intvl_stds*sqrt.(diag(Σ_centered)),
                         zorder = 0,
                         color=colour_pred_nearby, alpha=0.3)
    end

    ts_true = test_subset[:ts].values
    temp_true = test_subset[:temp].values
    if truth
        mean_true = mean(temp_true)
        plt.plot(ts_true, temp_true.-mean_true,
                 zorder=15,
                 marker="o",
                 markersize=markersize,
                 color=colour_truth, label="true hourly")
    end
    plt.gca()[:xaxis][:set_major_formatter](plt.matplotlib[:dates][:DateFormatter]("%Y-%m-%d"))
    # plt.xticks(collect(Date(xlim[1]):Day(1):Date(xlim[2])))
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

