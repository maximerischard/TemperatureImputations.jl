using DataTables
using CSV
using Base.Dates: Hour, Day, Date

function read_station(usaf::Int, wban::Int, id::Int; data_dir::String=".")
    fn = @sprintf("%d.%d.processed.2015.2015.csv", usaf, wban)
    station_data = CSV.read(join((data_dir, "/data2015/",fn)), DataTable,
                            datarow=1,
                            header=[:year, :month, :day, :hour, :min, :seconds, :temp])
    station_data[:temp][isnan(station_data[:temp].values)].isnull[:] = true
    station_data = station_data[!station_data[:temp].isnull & !isnan(station_data[:temp].values) ,:]    
    station_ts = DateTime[DateTime(
        get(r[:year]),
        get(r[:month]),
        get(r[:day]),
        get(r[:hour]),
        get(r[:min]),
        get(r[:seconds])
        ) for r in eachrow(station_data)]
    station_data[:ts] = station_ts
    station_data[:station] = id
    return station_data
end
function read_isdList(;data_dir::String=".")
    # Read stations data
    isdList = CSV.read(join((data_dir,"/isdList.csv")), DataTable)

    # Project onto Euclidean plane
    epsg=Proj4.Projection(Proj4.epsg[2794])
    wgs84=Proj4.Projection("+proj=longlat +datum=WGS84 +no_defs")
    isdProj = Proj4.transform(wgs84,epsg,convert(Matrix, isdList[[:LON,:LAT]]))
    isdList[:X_PRJ] = isdProj[:,1]
    isdList[:Y_PRJ] = isdProj[:,2]
    return isdList
end
function add_ts_hours!(df::DataTable)
    # timestamps in hours
    ms_per_hour = 1e3*3600
    ts_vec = convert(Vector{Float64}, df[:ts].values.-get(minimum(df[:ts]))) / ms_per_hour
    df[:ts_hours] = ts_vec
    return df
end
function read_Stations(isdSubset; data_dir::String=".")
    station_IDs = [(get(r[:USAF]), get(r[:WBAN])) for r in eachrow(isdSubset)]
    hourly_ls = [read_station(sid[1], sid[2], i; data_dir=data_dir) for (i,sid) in enumerate(station_IDs)]
    hourly_cat = vcat(hourly_ls)
    add_ts_hours!(hourly_cat)
    return hourly_cat
end
function test_data(hourly::DataTable, istation::Int, hr_measure::Hour)
    hourly_test = hourly[hourly[:station].values .== istation,:]
    hourly_test[:ts_day] = [measurement_date(t, hr_measure) for t in hourly_test[:ts].values]
    TnTx = DataTables.by(hourly_test, :ts_day, df -> DataTable(
        Tn=minimum(df[:temp].values), 
        Tx=maximum(df[:temp].values),
        Tn_time=df[:ts].values[indmin(df[:temp].values)],
        Tx_time=df[:ts].values[indmax(df[:temp].values)],
        times_p_day=nrow(df),
    ))
    return TnTx
end
