functions {
    real smoothmax(vector x, real k, real maxkx){
        return (maxkx+log(sum(exp(k*x - maxkx))))/k;
    }
    real smoothmin(vector x, real k, real minkx){
        return -smoothmax(-x, k, -minkx);
    }
}
data {
    int<lower=0> N; // number of observations
    real Xmax;
    real Xmin;
    real mu_i[N];
    real<lower=0> sigma_i[N];
    real<lower=0> k;
}
parameters {
    vector[N] X_i; // latent variables
}
transformed parameters {
    real Xsmoothmax;
    real Xsmoothmin;
    Xsmoothmax = smoothmax(X_i, k, k*Xmax);
    Xsmoothmin = smoothmin(X_i, k, k*Xmin);
}
model {
    X_i ~ normal(mu_i, sigma_i);
    Xmax ~ normal(Xsmoothmax, 0.01);
    Xmin ~ normal(Xsmoothmin, 0.01);
}
