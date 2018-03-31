using DataFrames
using CSV
using Base.Dates: Hour, Day, Millisecond, Date

function read_station(usaf::Int, wban::Int, id::Int; data_dir::String=".")
    fn = @sprintf("%06d.%05d.processed.2015.2015.csv", usaf, wban)
    station_data = CSV.read(joinpath(data_dir, "data2015", fn), DataFrame,
                            datarow=1,
                            header=[:year, :month, :day, :hour, :min, :seconds, :temp])
    # remove missing data (null or nan)
    station_data = station_data[.!isnan.(station_data[:temp]), :]
    DataFrames.dropmissing!(station_data)
    station_ts = DateTime[DateTime(
        r[:year],
        r[:month],
        r[:day],
        r[:hour],
        r[:min],
        r[:seconds]
        ) for r in DataFrames.eachrow(station_data)]
    station_data[:ts] = station_ts
    station_data[:station] = id
    return station_data
end
function read_isdList(;data_dir::String=".")
    # Read stations data
    # isdList = CSV.read(joinpath(data_dir,"isdList.csv"), DataFrame;
        # header=1,
        # weakrefstrings=false
        # )
    isdList = DataFrames.readtable(joinpath(data_dir,"isdList.csv"))

    # Project onto Euclidean plane
    epsg=Proj4.Projection(Proj4.epsg[2794])
    wgs84=Proj4.Projection("+proj=longlat +datum=WGS84 +no_defs")
    isdProj = Proj4.transform(wgs84,epsg, hcat(isdList[:LON], isdList[:LAT]))
    isdList[:X_PRJ] = isdProj[:,1]
    isdList[:Y_PRJ] = isdProj[:,2]
    return isdList
end
function add_ts_hours!(df::DataFrame)
    # timestamps in hours
    ts_vals = df[:ts]
    min_dt = minimum(ts_vals)
    ms_per_hour = convert(Millisecond, Hour(1))
    g_hours_from_min = (dt) -> convert(Millisecond, dt - min_dt) / ms_per_hour
    ts_vec = g_hours_from_min.(ts_vals)
    df[:ts_hours] = ts_vec
    return df
end
function read_Stations(isdSubset; data_dir::String=".")
    station_IDs = [(r[:USAF], r[:WBAN]) for r in DataFrames.eachrow(isdSubset)]
    hourly_ls = [read_station(sid[1], sid[2], i; data_dir=data_dir) for (i,sid) in enumerate(station_IDs)]
    hourly_cat = vcat(hourly_ls...)
    add_ts_hours!(hourly_cat)
    return hourly_cat
end
function test_data(hourly::DataFrame, istation::Int, hr_measure::Hour)
    hourly_test = hourly[hourly[:station] .== istation,:]
    hourly_test[:ts_day] = [measurement_date(t, hr_measure) for t in hourly_test[:ts]]
    TnTx = DataFrames.by(hourly_test, :ts_day, df -> DataFrame(
        Tn=minimum(df[:temp]), 
        Tx=maximum(df[:temp]),
        Tn_time=df[:ts][indmin(df[:temp])],
        Tx_time=df[:ts][indmax(df[:temp])],
        times_p_day=DataFrames.nrow(df),
    ))
    return TnTx
end
