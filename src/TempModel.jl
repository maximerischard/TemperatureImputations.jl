module TempModel
    using GaussianProcesses
    using PDMats
    using Optim
    # using Mamba
    using Stan
    import Proj4
    include("GPrealisations.jl")
    include("utils.jl")
    include("preprocessing.jl")
    include("predict_from_nearby.jl")
    include("stan_impute.jl")
    include("fitted_kernel.jl")
end
