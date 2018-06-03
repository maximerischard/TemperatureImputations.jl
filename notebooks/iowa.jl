data_dir = "../"
include(data_dir*"/src/TempModel.jl")

"""
    Convenience function for all notebooks to load and preprocess the Iowa data.
"""
function prepare_iowa_data(data_dir)
    # obtain the list of stations
    logging(DevNull; kind=:warn)
    isdList=TempModel.read_isdList(;data_dir=data_dir)
    # just the Iowa stations
    isdSubset=isdList[[(usaf in (725450,725460,725480,725485)) for usaf in isdList[:USAF]],:]

    # obtain the hourly temperature measurements for those stations
    hourly_data = TempModel.read_Stations(isdSubset; data_dir=data_dir)
    hourly_data[:hour] = round.(hourly_data[:ts], Dates.Hour)
    logging()

    # mark station 3 (Waterloo) as the test station
    itest=3
    # get the USAF code for the test station
    test_usaf=isdSubset[itest,:USAF]

    # emulate daily Tx/Tn measurements made at 17:00 UTC
    hr_measure = Hour(17)
    TnTx = TempModel.test_data(hourly_data, itest, hr_measure)

    hourly_train = hourly_data[hourly_data[:station].!=itest,:]
    hourly_test  = hourly_data[hourly_data[:station].==itest,:]

    hourly_test[:ts_day] = [TempModel.measurement_date(t, hr_measure) for t in hourly_test[:ts]]
    # add column to test data for TnTx (useful for plotting)
    hourly_test = join(hourly_test, TnTx, on=:ts_day)

    return Dict(
        :isdList => isdList,
        :isdSubset => isdSubset,
        :hourly_data => hourly_data,
        :itest => itest,
        :test_usaf => test_usaf,
        :TnTx => TnTx,
        :hourly_train => hourly_train,
        :hourly_test => hourly_test,
       )
end

function optim_kernel(k_spatiotemporal::Kernel, logNoise_init::Float64, stations_data::DataFrame, hourly_data::DataFrame, method::Symbol=:NLopt; 
                      x_tol=1e-5, f_tol=1e-10)
    chunks=GPE[]
    chunk_width=24*10 # 10 days at a time
    tstart=0.0
    nobsv=0
    max_time = maximum(hourly_data[:ts_hours])
    while tstart < max_time
        tend=tstart+chunk_width
        in_chunk= tstart .<= hourly_data[:ts_hours] .< tend
        hourly_chunk = hourly_data[in_chunk,:]
        nobsv_chunk = sum(in_chunk)
        nobsv += nobsv_chunk

        chunk_X_PRJ = stations_data[:X_PRJ][hourly_chunk[:station]]
        chunk_Y_PRJ = stations_data[:Y_PRJ][hourly_chunk[:station]]
        chunk_X = [hourly_chunk[:ts_hours] chunk_X_PRJ chunk_Y_PRJ]

        y = hourly_chunk[:temp]
        chunk = GPE(chunk_X', y, MeanConst(mean(y)), k_spatiotemporal, logNoise_init)
        push!(chunks, chunk)

        tstart=tend
    end
    reals = TempModel.GPRealisations(chunks)
    local min_neg_ll
    local min_hyp
    local opt_out
    if method == :NLopt
        min_neg_ll, min_hyp, ret, count = TempModel.optimize_NLopt(reals, domean=false, x_tol=x_tol, f_tol=f_tol)
        opt_out = (min_neg_ll, min_hyp, ret, count)
    elseif method == :Optim
        opt_out = TempModel.optimize!(reals, domean=false, 
                                      options=Optim.Options(x_tol=x_tol, f_tol=f_tol)
                                     )
        min_hyp = Optim.minimizer(opt_out)
        min_neg_ll = Optim.minimum(opt_out)
    else
        throw(MethodError())
    end
    @assert min_neg_ll ≈ -reals.mll
    return Dict(
        :hyp => min_hyp,
        :logNoise => reals.logNoise,
        :mll => -min_neg_ll,
        :opt_out => opt_out,
       )
end
