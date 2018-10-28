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
