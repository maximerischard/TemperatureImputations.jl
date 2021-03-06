function subset(df, from, to)
    after_from = df.ts .>= from
    before_to = df.ts .<= to
    return df[after_from .& before_to, :]
end

mutable struct NearbyPrediction
    ts::Vector{DateTime}
    μ::Vector{Float64}
    Σ::PDMat{Float64,Array{Float64,2}}
end

function add_diag!(mat::AbstractMatrix, a::Float64)
    for i in 1:size(mat,1)
        mat[i,i] += a
    end
    return mat
end
function add_diag!(Σ::PDMats.PDMat, a::Float64)
    mat = Σ.mat
    add_diag!(mat, a)
    copyto!(Σ.chol.factors, mat)
    # cholfact!(Σ.chol.factors, Symbol(Σ.chol.uplo))
    cholesky!(Hermitian(Σ.chol.factors, Symbol(Σ.chol.uplo)))
    # @assert maximum(abs, mat .- Matrix(Σ.chol)) < 1e-10
    return Σ
end

const KernelDict = Dict{String,KernelData}
function masked_empty(masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix)
    X1view = view(X1,masked.active_dims,:)
    X2view = view(X2,masked.active_dims,:)
	wrappeddata = EmptyData()
    return MaskedData(X1view, X2view, wrappeddata)
end
function emptydata_cache(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, cache::KernelDict=KernelDict())
    key = kernel_data_key(k, X1, X2)
    if typeof(k) <: Masked
        kdata = masked_empty(k, X1, X2)
    elseif typeof(k) <: PairKernel
        emptydata_cache(leftkern(k), X1, X2, cache)
        emptydata_cache(rightkern(k), X1, X2, cache)
        return cache
    elseif typeof(k) <: FixedKernel
        emptydata_cache(k.kernel, X1, X2, cache)
        return cache
    else
        kdata = EmptyData()
    end
    cache[key] = kdata
    return cache
end 

function predict_from_nearby(hourly_data::DataFrame, stationDF::DataFrame, 
        k::Kernel, logNoise::Float64, 
        target::Int, from::DateTime, to::DateTime)
    hourly_train = hourly_data[hourly_data.station.!=target,:]
    hourly_test  = hourly_data[hourly_data.station.==target,:]

    train_subset = subset(hourly_train,from,to)
    avgtemp=DataFrames.combine(DataFrames.groupby(train_subset, :station),
               df->DataFrame(avgtemp=mean(df.temp)))
    train_subset = DataFrames.leftjoin(train_subset, avgtemp, on=:station)

    train_X_PRJ = stationDF.X_PRJ[train_subset.station]
    train_Y_PRJ = stationDF.Y_PRJ[train_subset.station]
    train_X = [train_subset.ts_hours train_X_PRJ train_Y_PRJ]
    train_Y = train_subset.temp .- train_subset.avgtemp

    test_subset = subset(hourly_test,from,to)
    test_X_PRJ = stationDF.X_PRJ[test_subset.station]
    test_Y_PRJ = stationDF.Y_PRJ[test_subset.station]
    test_X = [test_subset.ts_hours test_X_PRJ test_Y_PRJ]

    @show nrow(train_subset)
    @show nrow(test_subset)

    X = Matrix(train_X')
    masked_cache = emptydata_cache(k, X, X)
    println("train kernel data")
    @time kdata = KernelData(k, X, X, masked_cache)
    println("train GP")
    @time train_GP = GPE(X, train_Y, MeanZero(), k, logNoise, kdata);
    println("predictions")
    Xtest = Matrix(test_X')
    @time μ_pred, Σ_pred = GaussianProcesses.predict_f(train_GP, Xtest; full_cov=true);
    println("add noise")
    add_diag!(Σ_pred, exp(2*logNoise))
    Σ_PD = PDMat(Σ_pred)
    return NearbyPrediction(test_subset.ts, μ_pred, Σ_PD)
end
