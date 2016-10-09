using DataFrames

function read_station(usaf::Int, wban::Int, id::Int)
    fn = @sprintf("%d.%d.processed.2015.2015.csv", usaf, wban)
    station_data = readtable("data2015/"fn, header=false, 
        names=[:year, :month, :day, :hour, :min, :seconds, :temp])
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
function read_isdList()
    # Read stations data
    isdList=readtable("isdList.csv")

    # Project onto Euclidean plane
    epsg=Proj4.Projection(Proj4.epsg[2794])
    wgs84=Proj4.Projection("+proj=longlat +datum=WGS84 +no_defs")
    isdProj = Proj4.transform(wgs84,epsg,convert(Matrix, isdList[[:LON,:LAT]]))
    isdList[:X_PRJ] = isdProj[:,1]
    isdList[:Y_PRJ] = isdProj[:,2]
    return isdList
end
function add_ts_hours!(df::DataFrame)
    # timestamps in hours
    ms_per_hour = 1e3*3600
    ts_vec = convert(Vector{Float64}, df[:ts].values.-get(minimum(df[:ts]))) / ms_per_hour
    df[:ts_hours] = ts_vec
    return df
end
function read_Stations(isdSubset)

    # Restrict to IOWA
    isdSubset=isdList[[(usaf in (725450,725460,725480,725485)) for usaf in isdList[:USAF].values],:]

    station_IDs = [(get(r[:USAF]), get(r[:WBAN])) for r in eachrow(isdSubset)]

    hourly_ls = [read_station(sid[1], sid[2], i) for (i,sid) in enumerate(station_IDs)]

    hourly_cat = vcat(hourly_ls)
    add_ts_hours!(hourly_cat)
    return hourly_cat
end
