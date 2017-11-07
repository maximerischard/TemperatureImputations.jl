using Base.Dates: Day, Hour

function prep_data(nearby_pred::NearbyPrediction, TnTx::DataTable, 
                date_start::Date, hr_measure::Hour, impute_window::Day)
    #=window_start = DateTime(date_start) + hr_measure - Day(1)=#
    #=window_end = window_start + impute_window=#
    date_end = date_start + impute_window - Day(1)
    in_window = [(date_start <= measurement_date(ts, hr_measure) <= date_end) for ts in nearby_pred.ts]
    ts_window = nearby_pred.ts[in_window]
    μ_window = nearby_pred.μ[in_window]
    Σ_window = PDMat(nearby_pred.Σ.mat[in_window,in_window])
    
    # for each 
    ts_window_day = [measurement_date(dt, hr_measure) for dt in ts_window]
    window_days = collect(date_start:Day(1):date_end)
    window_TnTx=TnTx[[d ∈ window_days for d in TnTx[:ts_day].values],:]
    day_impute = convert(Vector{Int}, ts_window_day .- minimum(ts_window_day))+1
    imputation_data = Dict(
        "N_TxTn" => nrow(window_TnTx),
        "Tn" => window_TnTx[:Tn].values,
        "Tx" => window_TnTx[:Tx].values,
        "Nimpt" => sum(in_window),
        "day_impute" => day_impute,
        "impt_times_p_day" => window_TnTx[:times_p_day].values,
        "predicted_mean" => μ_window,
        "predicted_cov" => Σ_window.mat,
        "predicted_cov_chol" => full(Σ_window.chol[:L]),
        "k_smoothmax" => 10.0,
    )
    return imputation_data, ts_window
end

function get_imputation_model()
    imputation_model = """
        functions {
            real smoothmax(vector x, real k, real maxkx){
                return (maxkx+log(sum(exp(k*x - maxkx))))/k;
            }
            real smoothmin(vector x, real k, real minkx){
                return -smoothmax(-x, k, -minkx);
            }
        }
        data {
            // Tn Tx data
            int<lower=1> N_TxTn; //
            vector[N_TxTn] Tx;
            vector[N_TxTn] Tn;

            // imputation points (for which we have )
            int<lower=1> Nimpt;
            int<lower=1,upper=N_TxTn> day_impute[Nimpt];
            // number of hours recorded within each day
            int<lower=1> impt_times_p_day[N_TxTn];

            // prior 
            vector[Nimpt] predicted_mean;
            matrix[Nimpt,Nimpt] predicted_cov;
            matrix[Nimpt,Nimpt] predicted_cov_chol;

            // control smooth max hardness
            real<lower=0> k_smoothmax;
        }
        parameters {
            vector[Nimpt] w_uncorr;
            real mu;
        }
        transformed parameters {
            vector[Nimpt] temp_impt;
            real Tsmoothmax[N_TxTn];
            real Tsmoothmin[N_TxTn];  
            temp_impt = mu + predicted_mean + predicted_cov_chol*w_uncorr;
            {
                int istart;
                istart = 1;
                for (i in 1:N_TxTn){
                    int ntimes;
                    ntimes = impt_times_p_day[i];
                    Tsmoothmin[i] = smoothmin(segment(temp_impt,istart,ntimes), 
                                              k_smoothmax, 
                                              k_smoothmax*Tn[i]);
                    Tsmoothmax[i] = smoothmax(segment(temp_impt,istart,ntimes), 
                                              k_smoothmax,
                                              k_smoothmax*Tx[i]);
                    istart = istart + ntimes;
                }
            }
        }
        model {
            w_uncorr ~ normal(0,1);
            mu ~ normal(0, 100.0);
            Tn ~ normal(Tsmoothmin, 0.1);
            Tx ~ normal(Tsmoothmax, 0.1);
        }
    """
    stanmodel = Stanmodel(name="imputation", model=imputation_model)
    return stanmodel
end

"""
 convenience function to extract the imputed temperatures
 from the STAN model object
"""
function get_temperatures(sim::Mamba.Chains)
    temp_varnames=[@sprintf("temp_impt.%d", i) for i in 1:imputation_data["Nimpt"]]
    temp_samples=getindex(sim, :, temp_varnames, :).value
    return temp_samples
end
