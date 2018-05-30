cbbPalette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
colour_hourly_nearby = cbbPalette[1]
colour_truth = "black"
colour_pred_nearby = cbbPalette[7]
colour_pred_tntx = cbbPalette[3]
colour_impt_nearby = cbbPalette[7]
colour_impt_tntx = cbbPalette[3]
colour_tn = cbbPalette[5]
colour_tx = "red"

function plot_imputations(ts, temp_impute, local_time; impt_indices=[250])
    imputed_10, imputed_90 = get_temp_percentiles(temp_impute)
    μ = get_temp_mean(temp_impute)
    
    plt.fill_between(local_time.(ts), imputed_10, imputed_90, 
        edgecolor="none",
        linewidth=0,
        color=colour_pred_tntx, alpha=0.3)
    for i in impt_indices
        plt.plot(local_time.(ts), temp_impute[i,:,1],
            linewidth=1,
            color=colour_impt_tntx)
    end
    plt.plot(local_time.(ts), μ, color=colour_pred_tntx, linewidth=2, 
             label=L"$\mathrm{T}_\mathrm{miss} \mid \mathrm{T}_\mathrm{nearby}, \mathrm{T}_{n}, \mathrm{T}_{x}$")
end
function plot_truth(
        test_data::DataFrame, 
        window::FittingWindow, 
        hr_measure::Hour,
        local_time::Function; 
        tntx::Bool=false, markersize=5)
    test_window = get_test_fw(test_data, window, hr_measure)
    ts = test_window[:ts]
    temp = test_window[:temp]
    Tn = test_window[:Tn]
    Tx = test_window[:Tx]
    plt.plot(local_time.(ts), temp,
        marker="o",
        markersize=markersize,
        color=colour_truth, label="true hourly")
    if tntx
        plt.step(local_time.(ts), Tn, where="pre",
            color=colour_tn, linewidth=3, label=L"$T_n$",
            zorder=-1)
        plt.step(local_time.(ts), Tx, where="pre",
            color=colour_tx, linewidth=3, label=L"$T_x$",
            zorder=-1)
    end
end
function plot_predictive(
        nearby_pred::TempModel.NearbyPrediction, 
        test_data, train_data, stations_metadata,
        local_time::Function,
        xlim::Tuple{DateTime,DateTime};
        truth::Bool=true,
        neighbours::Bool=true,
        mean_impt::Bool=true,
        imputations::Int=0,
        markersize=5
        )

    test_subset = subset(test_data, xlim[1], xlim[2])
    train_subset = subset(train_data, xlim[1], xlim[2])
    μ = nearby_pred.μ
    Σ = nearby_pred.Σ
    nobsv = length(μ)
    
    centering = eye(nobsv) .- (1.0/nobsv) .* ones(nobsv, nobsv)
    Σ_centered = centering * Σ.mat * centering
    distr = MultivariateNormal(μ, Σ)
    if neighbours
        markers = ["v", "1", "p", "s"]
        for (i,station) in enumerate(unique(train_subset[:station]))
            sdata = train_subset[train_subset[:station].==station,:]
            ts=sdata[:ts]
            plt.plot(local_time.(ts), sdata[:temp].-mean(sdata[:temp]), 
                     color=colour_hourly_nearby, 
                     marker = markers[i],
                     markersize=markersize,
                     label=stations_metadata[station,:ICAO],
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
        plt.plot(local_time.(ts), 
                 temp_sim.-mean_sim, 
                 color=colour_impt_nearby, linewidth=1,
                 label=label,
                 zorder=20)
    end
    if mean_impt
        plt.plot(local_time.(ts), μ-mean_μ, color=colour_pred_nearby, linewidth=2, 
                 zorder=30,
                 label=L"$\mathrm{T}_\mathrm{miss} \mid \mathrm{T}_\mathrm{nearby}$")

        intvl_width = 0.8
        intvl_stds = -quantile(Normal(0,1), (1-intvl_width)/2)

        plt.fill_between(local_time.(ts), 
                         μ-mean_μ-intvl_stds*sqrt.(diag(Σ_centered)),
                         μ-mean_μ+intvl_stds*sqrt.(diag(Σ_centered)),
                         zorder = 0,
                         edgecolor="none",
                         linewidth=0,
                         color=colour_pred_nearby, alpha=0.3)
    end

    ts_true = test_subset[:ts]
    temp_true = test_subset[:temp]
    if truth
        mean_true = mean(temp_true)
        plt.plot(local_time.(ts_true), temp_true.-mean_true,
                 zorder=15,
                 marker="o",
                 markersize=markersize,
                 color=colour_truth, label="true hourly")
    end
    plt.gca()[:xaxis][:set_major_formatter](plt.matplotlib[:dates][:DateFormatter]("%Y-%m-%d"))
    plt.xlim(local_time.(xlim))
end
function plot_residuals(nearby::TempModel.NearbyPrediction, test_data)
    ts = nearby.ts
    ts_start = minimum(ts)
    ts_end = maximum(ts)
    test_subset = subset(test_data, ts_start, ts_end)

    μ = nearby.μ
    Σ = nearby.Σ
    nobsv = length(μ)
    
    centering = eye(nobsv) .- (1.0/nobsv) .* ones(nobsv, nobsv)
    Σ_centered = centering * Σ.mat * centering
    distr = MultivariateNormal(μ, Σ)
    temp_true = test_subset[:temp]
    println("var(truth - predicted mean)=", var(μ .- temp_true))
    residuals = μ .- temp_true
    plt.plot(ts, abs.(residuals .- mean(residuals)))
    plt.ylabel("Absolute residuals")
end

