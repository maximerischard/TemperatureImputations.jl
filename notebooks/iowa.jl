import Dates
import TemperatureImputations

"""
    Convenience function for all notebooks to load and preprocess the Iowa data.
"""
function prepare_iowa_data(data_dir)
    # obtain the list of stations
    isdList=TemperatureImputations.read_isdList(;data_dir=data_dir, epsg=5072)
    # just the Iowa stations
    isdSubset=isdList[[(usaf in (725450,725460,725480,725485)) for usaf in isdList.USAF],:]

    # obtain the hourly temperature measurements for those stations
    hourly_data = TemperatureImputations.read_Stations(isdSubset; data_dir=data_dir)
    hourly_data[!,:hour] = round.(hourly_data.ts, Dates.Hour)

    # mark station 3 (Waterloo) as the test station
    itest=3
    # get the USAF code for the test station
    test_usaf=isdSubset[itest,:USAF]

    # emulate daily Tx/Tn measurements made at 17:00 UTC
    hr_measure = Hour(17)
    TnTx = TemperatureImputations.test_data(hourly_data, itest, hr_measure)

    hourly_train = hourly_data[hourly_data.station.!=itest,:]
    hourly_test  = hourly_data[hourly_data.station.==itest,:]

    hourly_test[!,:ts_day] = [TemperatureImputations.measurement_date(t, hr_measure) for t in hourly_test.ts]
    # add column to test data for TnTx (useful for plotting)
    hourly_test = leftjoin(hourly_test, TnTx, on=:ts_day)

    return Dict(
        :isdList => isdList,
        :isdSubset => isdSubset,
        :hourly_data => hourly_data,
        :itest => itest,
        :test_usaf => test_usaf,
        :TnTx => TnTx,
        :hourly_train => hourly_train,
        :hourly_test => hourly_test,
        :hr_measure => hr_measure,
       )
end

function prepare_ICAO_data(ICAO, data_dir; k_nearest=5)
    # obtain the list of stations
    isdList=TemperatureImputations.read_isdList(;data_dir=data_dir, epsg=5072)
    # just the Iowa stations
    isd_wData = TemperatureImputations.stations_with_data(isdList; data_dir=data_dir)
    test_station = isd_wData[isd_wData.ICAO.==ICAO, :]

    @assert nrow(test_station) == 1
    USAF = test_station[1, :USAF]
    WBAN = test_station[1, :WBAN]

    isd_nearest_and_test = TemperatureImputations.find_nearest(isd_wData, USAF, WBAN, k_nearest)
    # obtain the hourly temperature measurements for those stations
    hourly_data = TemperatureImputations.read_Stations(isd_nearest_and_test; data_dir=data_dir)

    itest=1

    # emulate daily Tx/Tn measurements made at 17:00 UTC
    hr_measure = Hour(17)
    TnTx = TemperatureImputations.test_data(hourly_data, itest, hr_measure)

    hourly_train = hourly_data[hourly_data.station.!=itest,:]
    hourly_test  = hourly_data[hourly_data.station.==itest,:]

    hourly_test[!,:ts_day] = [TemperatureImputations.measurement_date(t, hr_measure) for t in hourly_test.ts]
    # add column to test data for TnTx (useful for plotting)
    hourly_test = leftjoin(hourly_test, TnTx, on=:ts_day)

    return Dict(
        :isdList => isdList,
        :isdSubset => isd_nearest_and_test,
        :hourly_data => hourly_data,
        :itest => itest,
        :test_usaf => USAF,
        :test_icao => ICAO,
        :test_wban => WBAN,
        :TnTx => TnTx,
        :hourly_train => hourly_train,
        :hourly_test => hourly_test,
       )
end
