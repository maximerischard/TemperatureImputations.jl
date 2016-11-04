using GaussianProcesses: RQIso, SEIso, set_params!, Masked
# Temporal
ksp1 = SEIso(log(2*10^5), log(1.0))
ksp2 = SEIso(log(2*10^5), log(1.0))
ksp3 = SEIso(log(2*10^5), log(1.0))
ksp4 = SEIso(log(2*10^5), log(1.0))
ksp5 = SEIso(log(2*10^5), log(1.0))
ksp6 = SEIso(log(2*10^5), log(1.0))
k_means = SEIso(log(10^4), log(10.0))
k_spatiotemporal_3 = fix(Masked(k1, [1])) * fix(Masked(ksp1, [2,3]), :lσ) +
                     fix(Masked(k2, [1])) * fix(Masked(ksp2, [2,3]), :lσ) +
                     fix(Masked(k3, [1])) * fix(Masked(ksp3, [2,3]), :lσ) +
                     fix(Masked(k4, [1])) * fix(Masked(ksp4, [2,3]), :lσ) +
                     fix(Masked(k5, [1])) * fix(Masked(ksp5, [2,3]), :lσ) +
                     fix(Masked(k6, [1])) * fix(Masked(ksp6, [2,3]), :lσ) +
                     fix(Masked(k_means, [2,3]))
hyp = [14.9184,9.8588,10.6024,15.1699,13.6829,12.2061]
set_params!(k_spatiotemporal_3, hyp)
