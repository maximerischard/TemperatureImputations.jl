module TemperatureImputations
    using Statistics: mean

    using GaussianProcesses
    using GaussianProcesses: Mean, Kernel, evaluate, metric
    import GaussianProcesses: optimize!, get_optim_target
    import GaussianProcesses: num_params, set_params!, get_params, update_mll!
    import GaussianProcesses: update_mll_and_dmll!
    import GaussianProcesses: KernelData
    using GaussianProcesses: MaskedData, Masked, EmptyData, PairData, kernel_data_key, PairKernel, leftkern, rightkern, FixedKernel

    using PDMats
    using Optim
    using Optim: minimizer
    import StanBase, StanSample
    import Proj4
    import NLopt
    using Dates: Day, Hour, DateTime, Date, Millisecond, value
    using Printf: @sprintf
    using LinearAlgebra
    using LinearAlgebra: cholesky!, Hermitian
    import AxisArrays

    include("GPrealisations.jl")
    include("utils.jl")
    include("preprocessing.jl")
    include("predict_from_nearby.jl")
    include("stan_impute.jl")
    include("fitted_kernel.jl")
    include("variogram.jl")
    include("covariance.jl")
    include("smoothhmc.jl")
end # module
