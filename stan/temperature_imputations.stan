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
