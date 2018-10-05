function subset(df, from, to)
    after_from = df[:ts] .>= from
    before_to = df[:ts] .<= to
    return df[after_from .& before_to, :]
end

mutable struct NearbyPrediction
    ts::Vector{DateTime}
    μ::Vector{Float64}
    Σ::PDMat{Float64,Array{Float64,2}}
end

function add_diag!(Σ::PDMats.PDMat, a::Float64)
    mat = Σ.mat
    for i in 1:size(mat,1)
        mat[i,i] += a
    end
    copyto!(Σ.chol.factors, mat)
    # cholfact!(Σ.chol.factors, Symbol(Σ.chol.uplo))
    cholesky!(Hermitian(Σ.chol.factors, Symbol(Σ.chol.uplo)))
    @assert maximum(abs, mat .- Matrix(Σ.chol)) < 1e-10
    return Σ
end

function predict_from_nearby(hourly_data::DataFrame, stationDF::DataFrame, 
        k::Kernel, logNoise::Float64, 
        target::Int, from::DateTime, to::DateTime)
    hourly_train = hourly_data[hourly_data[:station].!=target,:]
    hourly_test  = hourly_data[hourly_data[:station].==target,:]

    train_subset = subset(hourly_train,from,to)
    avgtemp=DataFrames.by(train_subset, :station, 
               df->DataFrame(avgtemp=mean(df[:temp])))
    train_subset = join(train_subset, avgtemp, on=:station)

    train_X_PRJ = stationDF[:X_PRJ][train_subset[:station]]
    train_Y_PRJ = stationDF[:Y_PRJ][train_subset[:station]]
    train_X = [train_subset[:ts_hours] train_X_PRJ train_Y_PRJ]
    train_Y = train_subset[:temp] .- train_subset[:avgtemp]

    test_subset = subset(hourly_test,from,to)
    test_X_PRJ = stationDF[:X_PRJ][test_subset[:station]]
    test_Y_PRJ = stationDF[:Y_PRJ][test_subset[:station]]
    test_X = [test_subset[:ts_hours] test_X_PRJ test_Y_PRJ]

    train_GP = GP(train_X', train_Y, MeanZero(), k, logNoise);
    test_pred=GaussianProcesses.predict_f(train_GP, test_X'; full_cov=true);
    add_diag!(test_pred[2], exp(2*logNoise))
    return NearbyPrediction(test_subset[:ts], test_pred[1], test_pred[2])
end
