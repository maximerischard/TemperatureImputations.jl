using GaussianProcesses: Periodic, RQIso, SEIso, set_params!, Masked, fix

function fitted_temporal()
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
    logNoise = hyp[1]
    return k_time, logNoise
end

function fitted_sptemp_fixedvar(;kmean=true)
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

    k_spatiotemporal = fix(Masked(k1, [1])) * fix(Masked(ksp1, [2,3]), :lσ) +
                         fix(Masked(k2, [1])) * fix(Masked(ksp2, [2,3]), :lσ) +
                         fix(Masked(k3, [1])) * fix(Masked(ksp3, [2,3]), :lσ) +
                         fix(Masked(k4, [1])) * fix(Masked(ksp4, [2,3]), :lσ) +
                         fix(Masked(k5, [1])) * fix(Masked(ksp5, [2,3]), :lσ) +
                         fix(Masked(k6, [1])) * fix(Masked(ksp6, [2,3]), :lσ)
    if kmean
        k_means = SEIso(log(10^4), log(10.0))
        k_spatiotemporal = k_spatiotemporal + fix(Masked(k_means, [2,3]))
    end
    # parameters fitted in JuliaGP_spatial4.ipynb (I think)
    hyp = [-1.60354,15.4259,9.86874,9.12749,16.4496,15.0163,12.2061]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end

function fitted_sptemp_freevar(;kmean=true)
    k_time,_ = fitted_temporal()
    k1,k2,k3,k4,k5,k6=k_time.kerns
    ksp1 = SEIso(log(2*10^5), log(1.0))
    ksp2 = SEIso(log(2*10^5), log(1.0))
    ksp3 = SEIso(log(2*10^5), log(1.0))
    ksp4 = SEIso(log(2*10^5), log(1.0))
    ksp5 = SEIso(log(2*10^5), log(1.0))
    ksp6 = SEIso(log(2*10^5), log(1.0))
    k_spatiotemporal = fix(Masked(k1, [1])) * Masked(ksp1, [2,3]) +
                          fix(Masked(k2, [1])) * Masked(ksp2, [2,3]) +
                          fix(Masked(k3, [1])) * Masked(ksp3, [2,3]) +
                          fix(Masked(k4, [1])) * Masked(ksp4, [2,3]) +
                          fix(Masked(k5, [1])) * Masked(ksp5, [2,3]) +
                          fix(Masked(k6, [1])) * Masked(ksp6, [2,3])
    if kmean
        k_means = SEIso(log(10^4), log(10.0))
        k_spatiotemporal = k_spatiotemporal + fix(Masked(k_means, [2,3]))
    end
    hyp = [-1.65029,14.2398,0.111707,11.5002,-0.0791469,8.76624,0.126258,14.4041,0.147028,13.0326,-0.635492,12.2061,-8.08864e-7]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end

function fitted_sptemp_sumprod(;kmean=true)
    k1 = fix(Periodic(0.0,0.0,log(24.0)), :lp)
    k2 = RQIso(0.0,0.0,0.0)
    k3 = SEIso(0.0,0.0)
    k4 = RQIso(0.0,0.0,0.0)
    k5 = RQIso(0.0,0.0,0.0)
    k6 = SEIso(0.0,0.0)

    ksp1 = SEIso(log(2*10^5), log(1.0))
    ksp2 = SEIso(log(2*10^5), log(1.0))
    ksp3 = SEIso(log(2*10^5), log(1.0))
    ksp4 = SEIso(log(2*10^5), log(1.0))
    ksp5 = SEIso(log(2*10^5), log(1.0))
    ksp6 = SEIso(log(2*10^5), log(1.0))

    k_spatiotemporal = Masked(k1, [1]) * Masked(ksp1, [2,3]) +
                     Masked(k2, [1]) * Masked(ksp2, [2,3]) +
                     Masked(k3, [1]) * Masked(ksp3, [2,3]) +
                     Masked(k4, [1]) * Masked(ksp4, [2,3]) +
                     Masked(k5, [1]) * Masked(ksp5, [2,3]) +
                     Masked(k6, [1]) * Masked(ksp6, [2,3])
    if kmean
        k_means = SEIso(log(10^4), log(10.0))
        k_spatiotemporal = k_spatiotemporal + fix(Masked(k_means, [2,3]))
    end
    hyp = [-1.68206,-0.179317,0.945821,13.5116,0.0501475,0.866468,0.758593,-0.984024,11.0867,-0.38583,-1.44126,-1.13345,9.20607,0.0421904,2.12626,1.24119,-0.15271,15.081,0.129167,3.68457,0.701431,3.00982,14.0459,-1.5127,7.70676,-5.30466,12.2061,-6.18869e-6]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end

function fitted_sptemp_SExSE(;kmean=true)
    k_time = SEIso(0.0,0.0)
    k_spatial = fix(SEIso(log(2*10^5), log(1.0)), :lσ)
    k_spatiotemporal = Masked(k_time, [1]) * Masked(k_spatial, [2,3])
    if kmean
        k_means = SEIso(log(10^4), log(10.0))
        k_spatiotemporal = k_spatiotemporal + fix(Masked(k_means, [2,3]))
    end
    hyp = [-0.822261,0.996834,1.3172,12.0805]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1] 
    return k_spatiotemporal, logNoise
end

function fitted_sptemp_diurnal(;kmean=true)
    k_time = SEIso(0.0,0.0)
    k_spatial = fix(SEIso(log(2*10^5), log(1.0)), :lσ)
    k_periodic = fix(Periodic(log(1.0), -1.0, log(24.0)), :lp)
    k_diurndecay = fix(SEIso(log(10^5), 0.0), :lσ)
    k_spatiotemporal = Masked(k_time, [1]) * Masked(k_spatial, [2,3]) + 
        Masked(k_periodic, [1]) * Masked(k_diurndecay, [2,3])
    if kmean
        k_means = SEIso(log(10^4), log(10.0))
        k_spatiotemporal = k_spatiotemporal + fix(Masked(k_means, [2,3]))
    end
    hyp = [-0.82337,1.02776,1.14186,11.9454,-0.383965,0.858384,14.1618]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end

function fitted_sptemp_simpler(;kmean=true)
    k1 = fix(Periodic(0.0,0.0,log(24.0)), :lp)
    k2 = RQIso(0.0,0.0,0.0)
    k3b = RQIso(0.0,0.0,0.0)
    k4 = RQIso(0.0,0.0,0.0)

    ksp1 = SEIso(log(2*10^5), log(1.0))
    ksp2 = SEIso(log(2*10^5), log(1.0))
    ksp3 = SEIso(log(2*10^5), log(1.0))
    ksp4 = SEIso(log(2*10^5), log(1.0))

    k_spatiotemporal = Masked(k1, [1]) * Masked(ksp1, [2,3]) +
                     Masked(k2, [1])  * Masked(ksp2, [2,3]) +
                     Masked(k3b, [1]) * Masked(ksp3, [2,3]) +
                     Masked(k4, [1])  * Masked(ksp4, [2,3])
    if kmean
        k_means = SEIso(log(10^4), log(10.0))
        k_spatiotemporal = k_spatiotemporal + fix(Masked(k_means, [2,3]))
    end
    # logNoise TBD
    hyp = [-1.72527,-0.210656,0.950124,13.574,0.0544504,0.653978,0.538881,0.125985,10.9932,-0.605542,-1.2208,-0.946048,-1.0721,9.20607,0.229593,2.18355,1.29463,-1.14171,12.8224,0.182608]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end

