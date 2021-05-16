using GaussianProcesses: Periodic, RQIso, SEIso, set_params!, Masked, fix
using Printf

const SPACEDIM = [2,3]
const TIMEDIM  = [1]
const DEFAULT_LENGTHSCALE = 5e4

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

kprod(ktime::Kernel, kspace::Kernel) = Masked(ktime, TIMEDIM) * Masked(fix(kspace, :lσ), SPACEDIM)
kprod(ktime::Kernel, kspace::Kernel, knoise::Noise) = Masked(fix(ktime, :lσ), TIMEDIM) * 
                                                      (Masked(kspace, SPACEDIM) + Masked(knoise, SPACEDIM))
add_kmean(kspatiotemporal::Kernel, k_means::Noise) = kspatiotemporal + fix(Masked(k_means, SPACEDIM))
function fitted_sptemp_fixedvar(;kmean::Bool)
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

    ksp1 = SEIso(log(2*DEFAULT_LENGTHSCALE), log(1.0))
    ksp2 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp3 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp4 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp5 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp6 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))

    k_spatiotemporal = fix(Masked(k1, TIMEDIM)) * fix(Masked(ksp1, SPACEDIM), :lσ) +
                         fix(Masked(k2, TIMEDIM)) * fix(Masked(ksp2, SPACEDIM), :lσ) +
                         fix(Masked(k3, TIMEDIM)) * fix(Masked(ksp3, SPACEDIM), :lσ) +
                         fix(Masked(k4, TIMEDIM)) * fix(Masked(ksp4, SPACEDIM), :lσ) +
                         fix(Masked(k5, TIMEDIM)) * fix(Masked(ksp5, SPACEDIM), :lσ) +
                         fix(Masked(k6, TIMEDIM)) * fix(Masked(ksp6, SPACEDIM), :lσ)
    k_means = Noise(log(40.0))
    if kmean
        k_spatiotemporal = add_kmean(k_spatiotemporal, k_means)
    end
    # parameters fitted in JuliaGP_spatial4.ipynb (I think)
    hyp = [-1.60354,15.4259,9.86874,9.12749,16.4496,15.0163,12.2061]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end

function fitted_sptemp_freevar(;kmean::Bool)
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
    ksp1 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp2 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp3 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp4 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp5 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp6 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    k_spatiotemporal = fix(Masked(k1, TIMEDIM)) * Masked(ksp1, SPACEDIM) +
                          fix(Masked(k2, TIMEDIM)) * Masked(ksp2, SPACEDIM) +
                          fix(Masked(k3, TIMEDIM)) * Masked(ksp3, SPACEDIM) +
                          fix(Masked(k4, TIMEDIM)) * Masked(ksp4, SPACEDIM) +
                          fix(Masked(k5, TIMEDIM)) * Masked(ksp5, SPACEDIM) +
                          fix(Masked(k6, TIMEDIM)) * Masked(ksp6, SPACEDIM)
    k_means = Noise(log(40.0))
    if kmean
        k_spatiotemporal = add_kmean(k_spatiotemporal, k_means)
    end
    hyp = [-1.65029,14.2398,0.111707,11.5002,-0.0791469,8.76624,0.126258,14.4041,0.147028,13.0326,-0.635492,12.2061,-8.08864e-7]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end

function fitted_sptemp_sumprod(;kmean::Bool)
    k1 = fix(Periodic(0.0,0.0,log(24.0)), :lp)
    k2 = RQIso(0.0,0.0,0.0)
    k3 = SEIso(0.0,0.0)
    k4 = RQIso(0.0,0.0,0.0)
    k5 = RQIso(0.0,0.0,0.0)
    k6 = SEIso(0.0,0.0)

    ksp1 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp2 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp3 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp4 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp5 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp6 = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))

    k_spatiotemporal = Masked(k1, TIMEDIM) * Masked(ksp1, SPACEDIM) +
                     Masked(k2, TIMEDIM) * Masked(ksp2, SPACEDIM) +
                     Masked(k3, TIMEDIM) * Masked(ksp3, SPACEDIM) +
                     Masked(k4, TIMEDIM) * Masked(ksp4, SPACEDIM) +
                     Masked(k5, TIMEDIM) * Masked(ksp5, SPACEDIM) +
                     Masked(k6, TIMEDIM) * Masked(ksp6, SPACEDIM)
    k_means = Noise(log(40.0))
    if kmean
        k_spatiotemporal = add_kmean(k_spatiotemporal, k_means)
    end
    hyp = [-1.68206,-0.179317,0.945821,13.5116,0.0501475,0.866468,0.758593,-0.984024,11.0867,-0.38583,-1.44126,-1.13345,9.20607,0.0421904,2.12626,1.24119,-0.15271,15.081,0.129167,3.68457,0.701431,3.00982,14.0459,-1.5127,7.70676,-5.30466,12.2061,-6.18869e-6]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end
function kernel_sptemp_SExSE(;kmean::Bool)
    k_time = SEIso(0.0, 0.0)
    k_spatial = SEIso(log(DEFAULT_LENGTHSCALE), log(1.0))
    k_spatiotemporal = kprod(k_time, k_spatial)
    k_means = Noise(log(40.0))
    if kmean
        k_spatiotemporal = add_kmean(k_spatiotemporal, k_means)
    end
    return Dict(
        :time=>k_time,
        :space=>k_spatial,
        :mean=>k_means,
        :spatiotemporal => k_spatiotemporal
        )
end
function showkernel_SExSE(kdict::Dict, logNoise)
    k_time, k_spatial = kdict[:time], kdict[:space]
    print("\nk: Temporal kernel \n=================\n")
    @printf("σ: %5.3f\n", √k_time.σ2)
    @printf("l: %5.3f hours\n", √k_time.ℓ2)
    print("\nk: Spatial kernel \n=================\n")
    @printf("σ: %5.3f\n", √k_spatial.σ2)
    @printf("l: %5.3f km\n", √k_spatial.ℓ2 / 1000)
    print("\n=================\n")
    @printf("σy: %5.3f\n", exp(logNoise))
end
function fitted_sptemp_SExSE(;kmean::Bool)
	kdict = kernel_sptemp_SExSE(;kmean=kmean)
	k_spatiotemporal = kdict[:spatiotemporal]
    hyp = [-0.822261,0.996834,1.3172,12.0805]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1] 
    return k_spatiotemporal, logNoise
end

function kernel_sptemp_diurnal(;kmean::Bool)
    k_time = SEIso(0.0,0.0)
    k_spatial = Mat32Iso(log(DEFAULT_LENGTHSCALE), log(1.0))
    k_periodic = Periodic(log(1.0), log(3.0), log(24.0))
    k_diurndecay = SEIso(log(DEFAULT_LENGTHSCALE), 0.0)
    k_spatiotemporal = kprod(k_time, k_spatial) + 
                       kprod(fix(k_periodic, :lp), k_diurndecay)
    k_means = Noise(log(40.0))
    if kmean
        k_spatiotemporal = add_kmean(k_spatiotemporal, k_means)
    end
    return Dict(
        :time=>k_time,
        :space=>k_spatial,
        :diurnal=>k_periodic,
        :diurndecay=>k_diurndecay,
        :mean=>k_means,
        :spatiotemporal => k_spatiotemporal
        )
end
function fitted_sptemp_diurnal(;kmean::Bool)
	kdict = kernel_sptemp_diurnal(;kmean=kmean)
	k_spatiotemporal = kdict[:spatiotemporal]
    hyp = [-0.82337,1.02776,1.14186,11.9454,-0.383965,0.858384,14.1618]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end

function kernel_sptemp_matern(;kmean::Bool)
    kt1 = Periodic(log(1.0),log(3.0),log(24.0))
    kt2 = RQIso(log(0.5),0.0,0.0)  # half an hour
    kt3 = RQIso(log(2.0),0.0,0.0)  # two hours
    kt4 = RQIso(log(12.0),0.0,0.0) # twelve hours

    ksp1 = Mat32Iso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp2 = Mat32Iso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp3 = Mat32Iso(log(DEFAULT_LENGTHSCALE), log(1.0))
    ksp4 = Mat32Iso(log(DEFAULT_LENGTHSCALE), log(1.0))
    
    k_means = Noise(log(40.0))
    

    k_spatiotemporal = kprod(fix(kt1, :lp), ksp1) +
                       kprod(kt2, ksp2) +
                       kprod(kt3, ksp3) +
                       kprod(kt4, ksp4)
    if kmean
        k_spatiotemporal = add_kmean(k_spatiotemporal, k_means)
    end
    return Dict(
        :time1=>kt1, :time2=>kt2, :time3=>kt3, :time4=>kt4,
        :space1=>ksp1, :space2=>ksp2, :space3=>ksp3, :space4=>ksp4,
        :mean=>k_means,
        :spatiotemporal => k_spatiotemporal
        )
end
function kernel_sptemp_maternlocal(;kmean::Bool)
    kt1 = Periodic(log(1.0),log(3.0),log(24.0))
    # start with a fairly high α
    kt2 = RQIso(log(0.5),  log(1.0), log(5.0))  # half an hour
    kt3 = RQIso(log(2.0),  log(1.0), log(5.0))  # two hours
    kt4 = RQIso(log(12.0), log(1.0), log(5.0)) # twelve hours

    ksp1 = Mat32Iso(log(5e4), log(1.0))
    ksp2 = Mat32Iso(log(5e4), log(1.0))
    ksp3 = Mat32Iso(log(5e4), log(1.0))
    ksp4 = Mat32Iso(log(5e4), log(1.0))

    kn1 = Noise(log(1.0))
    kn2 = Noise(log(1.0))
    kn3 = Noise(log(1.0))
    kn4 = Noise(log(1.0))
    
    k_means = Noise(log(40.0))
    

    k_spatiotemporal = kprod(fix(kt1, :lp), ksp1, kn1) +
                       kprod(kt2, ksp2, kn2) +
                       kprod(kt3, ksp3, kn3) +
                       kprod(kt4, ksp4, kn4)
    if kmean
        k_spatiotemporal = add_kmean(k_spatiotemporal, k_means)
    end
    return Dict(
        :time1=>kt1, :time2=>kt2, :time3=>kt3, :time4=>kt4,
        :space1=>ksp1, :space2=>ksp2, :space3=>ksp3, :space4=>ksp4,
        :noise1=>kn1, :noise2=>kn2, :noise3=>kn3, :noise4=>kn4,
        :mean=>k_means,
        :spatiotemporal => k_spatiotemporal
        )
end
function fitted_sptemp_matern(;kmean::Bool)
    kdict = kernel_sptemp_matern(kmean=kmean)
    k_spatiotemporal = kdict[:spatiotemporal]
    # hyperparameters fitted in `FitGP_spatiotemp_spatial_matern.ipynb`:
    hyp = [-1.72361, -0.200613, 1.0075, 14.0162, -1.20089, -0.703055, -1.07354, 9.35237, 0.678935, -0.0278533, 0.244548, 11.1962, 2.18646, 1.45579, -0.924148, 13.3052]
    set_params!(k_spatiotemporal, hyp[2:end])
    logNoise=hyp[1]
    return k_spatiotemporal, logNoise
end
