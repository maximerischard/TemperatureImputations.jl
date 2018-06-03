data {
    int<lower=0> N; // number of observations
    real Xmax;
    real Xmin;
    vector[N] mu_i;
    real<lower=0> sigma_i[N];
}
parameters {
    vector[N] X_i; // latent variables
}
model {
    X_i ~ normal(mu_i, sigma_i);
    Xmax ~ normal(max(X_i), 0.01);
    Xmin ~ normal(min(X_i), 0.01);
}
