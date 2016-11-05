using GaussianProcesses: RQIso, SEIso, set_params!, Masked
# Temporal
k1 = fix(Periodic(0.0,0.0,log(24.0)), :lp)
k2 = RQIso(0.0,0.0,0.0)
k3 = SEIso(0.0,0.0)
k4 = RQIso(0.0,0.0,0.0)
k5 = RQIso(0.0,0.0,0.0)
k6 = SEIso(0.0,0.0)
k_time=k1+k2+k3+k4+k5+k6
# hyperparameters fitted in JuliaGP_timeseries_chunks.ipynb
hyp=[-1.46229,-0.0777809,1.03854,1.45757,1.06292,-1.23699,-1.2353,-1.05117,3.10614,1.29327,2.84729,3.67464,0.537794,3.0094,7.70676,-5.30466]

set_params!(k_time, hyp[2:end])
ksp1 = SEIso(log(2*10^5), log(1.0))
ksp2 = SEIso(log(2*10^5), log(1.0))
ksp3 = SEIso(log(2*10^5), log(1.0))
ksp4 = SEIso(log(2*10^5), log(1.0))
ksp5 = SEIso(log(2*10^5), log(1.0))
ksp6 = SEIso(log(2*10^5), log(1.0))
k_means = SEIso(log(10^4), log(10.0))
k_spatiotemporal = fix(Masked(k1, [1])) * fix(Masked(ksp1, [2,3]), :lσ) +
                     fix(Masked(k2, [1])) * fix(Masked(ksp2, [2,3]), :lσ) +
                     fix(Masked(k3, [1])) * fix(Masked(ksp3, [2,3]), :lσ) +
                     fix(Masked(k4, [1])) * fix(Masked(ksp4, [2,3]), :lσ) +
                     fix(Masked(k5, [1])) * fix(Masked(ksp5, [2,3]), :lσ) +
                     fix(Masked(k6, [1])) * fix(Masked(ksp6, [2,3]), :lσ) +
                     fix(Masked(k_means, [2,3]))
hyp = [-1.59982,14.9184,9.8588,10.6024,15.1699,13.6829,12.2061]
set_params!(k_spatiotemporal, hyp[2:end])
logNoise=hyp[1]
