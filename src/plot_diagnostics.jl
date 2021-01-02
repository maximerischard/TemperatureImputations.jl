cbbPalette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
colour_hourly_nearby = cbbPalette[1]
colour_truth = "black"
colour_pred_nearby = cbbPalette[7]
colour_pred_tntx = cbbPalette[3]
colour_impt_nearby = cbbPalette[7]
colour_impt_tntx = cbbPalette[3]
colour_tn = cbbPalette[5]
colour_tx = "red"

function plot_imputations(ts, temp_impute, local_time; plot_mean=true, impt_indices=[250])
    imputed_10, imputed_90 = get_temp_percentiles(temp_impute)
    μ = get_temp_mean(temp_impute)
    label = L"$\mathrm{T}_\mathrm{miss} \mid \mathrm{T}_\mathrm{nearby}, \mathrm{T}_{n}, \mathrm{T}_{x}$"
    
    plt.fill_between(local_time.(ts), imputed_10, imputed_90, 
        edgecolor="none",
        linewidth=0,
        label = plot_mean ? "" : label,
        color=colour_pred_tntx, alpha=0.3)
    for i in impt_indices
        plt.plot(local_time.(ts), temp_impute[i,:,1],
            linewidth=1,
            color=colour_impt_tntx)
    end
    if plot_mean
        plt.plot(local_time.(ts), μ, color=colour_pred_tntx, linewidth=2, label=label)
    end
end

function plot_TnTx(hourly_data, station, hr_measure, local_time::Function; linewidth=3, zorder=-1, kwargs...)
    hourly_test = hourly_data[hourly_data[:station] .== station, :]
    TnTx = TemperatureImputations.test_data(hourly_test, station, hr_measure)
    hourly_test[:ts_day] = TemperatureImputations.measurement_date.(hourly_test[:ts], hr_measure)
    hourly_TnTx = join(hourly_test, TnTx, on=:ts_day)
    local_ts = local_time.(hourly_TnTx[:ts])
    plt.plot(local_ts, hourly_TnTx[:Tn], # where="pre",
             color=colour_tn, linewidth=linewidth, zorder=zorder, label=L"$T_n$"; kwargs...)
    plt.plot(local_ts, hourly_TnTx[:Tx], # where="pre",
             color=colour_tx, linewidth=linewidth, zorder=zorder, label=L"$T_x$"; kwargs...)
end

function plot_truth(
        test_data::DataFrame, 
        window::FittingWindow, 
        hr_measure::Hour,
        local_time::Function; 
        markersize=5)
    test_window = get_test_fw(test_data, window, hr_measure)
    ts = test_window[:ts]
    temp = test_window[:temp]
    plt.plot(local_time.(ts), temp,
        marker="o",
        markersize=markersize,
        color=colour_truth, label="true hourly")
end
function plot_neighbours(train_data, stations_metadata, local_time::Function, xlim::Tuple{DateTime,DateTime}; subtractmean::Bool, kwargs...)
    train_subset = subset(train_data, xlim[1], xlim[2])
    markers = ["v", "1", "p", "s", "X"]
    stations = unique(train_subset[:station])
    for (i,station) in enumerate(stations)
        sdata = train_subset[train_subset[:station].==station,:]
        ts = sdata[:ts]
        y = sdata[:temp]
        if subtractmean
            y .-= mean(temp)
        end
        label = stations_metadata[station, :ICAO]
        plt.plot(local_time.(ts), y;
                 marker = markers[i],
                 label=label,
                 kwargs...
                 )
    end
end
function plot_predictive(
        nearby_pred::TemperatureImputations.NearbyPrediction, 
        test_data, train_data, stations_metadata,
        local_time::Function,
        xlim::Tuple{DateTime,DateTime};
        truth::Bool=true,
        # neighbours::Bool=true, # use plot_neighbours instead
        mean_impt::Bool=true,
        imputations::Int=0,
        markersize=5,
        intvl_width = 0.8,
        subtractmean::Bool,
        cheat_mean::Bool
        )

    test_subset = subset(test_data, xlim[1], xlim[2])
    train_subset = subset(train_data, xlim[1], xlim[2])
    μ = nearby_pred.μ
    Σ = nearby_pred.Σ
    nobsv = length(μ)
    
    distr = MultivariateNormal(μ, Σ)
    ts=nearby_pred.ts
    centering = Matrix(1.0I, nobsv, nobsv) .- (1.0/nobsv)
    Σ_centered = centering * Σ.mat * centering
    in_window = (xlim[1] .<= ts) .& (ts .<= xlim[2])
    temp_true = test_subset[:temp]
    mean_true = mean(temp_true)
    for i in 1:imputations
        temp_sim = rand(distr)
        y = temp_sim[in_window]
        if subtractmean
            y .-= mean(y)
        elseif cheat_mean
            y .+= mean_true - mean(y)
        end
        mean_sim = mean(temp_sim[in_window])
        label = ""
        # if i==1
            # label = "example imputation"
        # end
        plt.plot(local_time.(ts[in_window]), 
                 y, 
                 color=colour_impt_nearby, linewidth=1,
                 label=label,
                 zorder=20)
    end
    if mean_impt
        y = μ[in_window]
        if subtractmean
            y .-= mean(y)
        elseif cheat_mean
            y .+= mean_true - mean(y)
        end
        plt.plot(local_time.(ts[in_window]), y, color=colour_pred_nearby, linewidth=2, 
                 zorder=30,
                 label=L"$\mathrm{T}_\mathrm{miss} \mid \mathrm{T}_\mathrm{nearby}$")

        intvl_stds = -quantile(Normal(0,1), (1-intvl_width)/2)

        plt.fill_between(local_time.(ts[in_window]), 
                         y.-intvl_stds.*sqrt.(diag(Σ_centered)[in_window]),
                         y.+intvl_stds.*sqrt.(diag(Σ_centered)[in_window]),
                         zorder = 0,
                         edgecolor="none",
                         linewidth=0,
                         color=colour_pred_nearby, alpha=0.3)
    end

    ts_true = test_subset[:ts]
    temp_true = test_subset[:temp]
    if truth
        y = temp_true
        if subtractmean
            y .-= mean(y)
        end
        plt.plot(local_time.(ts_true), y,
                 zorder=15,
                 marker="o",
                 markersize=markersize,
                 color=colour_truth, label="true hourly")
    end
    plt.gca()[:xaxis][:set_major_formatter](plt.matplotlib[:dates][:DateFormatter]("%Y-%m-%d"))
    plt.gcf()[:autofmt_xdate]()
    plt.xlim(local_time.(xlim))
end
function plot_residuals(nearby::TemperatureImputations.NearbyPrediction, test_data)
    ts = nearby.ts
    ts_start = minimum(ts)
    ts_end = maximum(ts)
    test_subset = subset(test_data, ts_start, ts_end)

    μ = nearby.μ
    Σ = nearby.Σ
    nobsv = length(μ)
    
    centering = Matrix(1.0I, nobsv, nobsv) .- (1.0/nobsv)
    Σ_centered = centering * Σ.mat * centering
    distr = MultivariateNormal(μ, Σ)
    temp_true = test_subset[:temp]
    println("var(truth - predicted mean)=", var(μ .- temp_true))
    residuals = μ .- temp_true
    plt.plot(ts, abs.(residuals .- mean(residuals)))
    plt.ylabel("Absolute residuals")
end

