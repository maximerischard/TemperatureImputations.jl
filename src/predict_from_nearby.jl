function subset(df, from, to)
    after_from = df[:ts].values .>= from
    before_to = df[:ts].values .<= to
    return df[after_from & before_to,:]
end

type NearbyPrediction
    ts::Vector{DateTime}
    μ::Vector{Float64}
    Σ::PDMat{Float64,Array{Float64,2}}
end

function predict_from_nearby(hourly_data::DataFrame, stationDF::DataFrame, 
        k::Kernel, logNoise::Float64, 
        target::Int, from::DateTime, to::DateTime)
    hourly_train = hourly_data[hourly_data[:station].values.!=target,:]
    hourly_test  = hourly_data[hourly_data[:station].values.==target,:]

    train_subset = subset(hourly_train,from,to)
    avgtemp=by(train_subset, :station, df->DataFrame(avgtemp=mean(df[:temp].values)))
    train_subset = join(train_subset, avgtemp, on=:station)

    train_X_PRJ = stationDF[:X_PRJ].values[train_subset[:station].values]
    train_Y_PRJ = stationDF[:Y_PRJ].values[train_subset[:station].values]
    train_X = [train_subset[:ts_hours].values train_X_PRJ train_Y_PRJ]
    train_Y = train_subset[:temp].values .- train_subset[:avgtemp].values

    test_subset = subset(hourly_test,from,to)
    test_X_PRJ = stationDF[:X_PRJ].values[test_subset[:station].values]
    test_Y_PRJ = stationDF[:Y_PRJ].values[test_subset[:station].values]
    test_X = [test_subset[:ts_hours].values test_X_PRJ test_Y_PRJ]

    train_GP = GP(train_X', train_Y, MeanZero(), k, logNoise);
    test_pred=predict(train_GP, test_X'; full_cov=true);
    return NearbyPrediction(test_subset[:ts].values, test_pred[1], test_pred[2])
end