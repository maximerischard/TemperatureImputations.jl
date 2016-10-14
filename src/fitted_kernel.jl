using GaussianProcesses: FixedPeriodic, RQIso, SEIso, set_params!, Masked
# Temporal
k1 = FixedPeriodic(0.0,0.0,log(24.0))
k2 = RQIso(0.0,0.0,0.0)
k3 = SEIso(0.0,0.0)
k4 = RQIso(0.0,0.0,0.0)
k5 = RQIso(0.0,0.0,0.0)
k6 = SEIso(0.0,0.0)
k_time=k1+k2+k3+k4+k5+k6
# hyperparameters fitted in JuliaGP_timeseries_chunks.ipynb
hyp=[-1.4693,-0.0806483,1.0449,1.50786,1.10795,-1.38548,-1.22736,-1.05138,3.09723,1.28737,2.84127,3.64666,0.469691,3.00962,7.70695,-5.39838]
set_params!(k_time, hyp[2:end])

# Spatial
k_spatial = SEIso(log(2*10^5), log(1.0))
hyp=[-1.5875,11.2445,0.132749]
set_params!(k_spatial, hyp[2:3])
logNoise=hyp[1]
k_spatiotemporal = Masked(k_time, [1]) * Masked(k_spatial, [2,3])
