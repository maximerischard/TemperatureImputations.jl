using DataFrames
using CSV

function read_station(usaf::Int, wban::Int, id::Int; data_dir::String=".")
    fn = @sprintf("%06d.%05d.processed.2015.2015.csv", usaf, wban)
    station_data = CSV.read(joinpath(data_dir, "data2015", fn), DataFrame,
                            datarow=1, copycols=true,
                            header=[:year, :month, :day, :hour, :min, :seconds, :temp],
                            types=Dict(:year=>Int64, :month=>Int64, :day=>Int64, 
                                       :hour=>Int64, :min=>Int64, :seconds=>Int64,
                                       :temp=>Float64),
                            )
    DataFrames.dropmissing!(station_data, disallowmissing=true)
    # remove missing data (null or nan)
    station_data = station_data[.!ismissing.(station_data.temp), :]
    # station_data = station_data[.!isnan.(station_data[:temp]), :]
    station_data = station_data[isfinite.(station_data.temp), :]
    # station_data[:temp] = Float64.(station_data[:temp])
    station_ts = DateTime[DateTime(r[:year], r[:month], r[:day],
                                   r[:hour], r[:min], r[:seconds]) 
                          for r in DataFrames.eachrow(station_data)]
    station_data[!,:ts] = station_ts
    station_data[!,:station] .= id
    return station_data
end
function read_isdList(;data_dir::String=".", epsg::Int)
    # Read stations data
    # isdList = CSV.read(joinpath(data_dir,"isdList.csv");
        # header=1,
        # weakrefstrings=false
        # )
    isdList = CSV.read(joinpath(data_dir,"isdList.csv"), DataFrame, copycols=true)

    # Project onto Euclidean plane
    epsg=Proj4.Projection(Proj4.epsg[epsg])
    wgs84=Proj4.Projection("+proj=longlat +datum=WGS84 +no_defs")
    isdProj = Proj4.transform(wgs84, epsg, hcat(isdList.LON, isdList.LAT))
    isdList[!,:X_PRJ] = isdProj[:,1]
    isdList[!,:Y_PRJ] = isdProj[:,2]
    return isdList
end
function stations_with_data(isdList::DataFrame; data_dir::String=".")
    data_files = readdir(joinpath(data_dir, "data2015"))
    data_files = filter(s -> endswith(s, ".csv"), data_files)
    data_USAF_WBAN = [parse.(Int, split(s, ".")[1:2]) for s in data_files]
    isd_wData = filter(row -> [row[:USAF], row[:WBAN]] ∈ data_USAF_WBAN, isdList)
    return isd_wData
end
function add_ts_hours!(df::DataFrame)
    # timestamps in hours
    ts_vals = df.ts
    min_dt = minimum(ts_vals)
    ms_per_hour = convert(Millisecond, Hour(1))
    g_hours_from_min = (dt) -> convert(Millisecond, dt - min_dt) / ms_per_hour
    ts_vec = g_hours_from_min.(ts_vals)
    df[!,:ts_hours] = ts_vec
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
    hourly_test = hourly[hourly.station .== istation,:]
    hourly_test.ts_day .= measurement_date.(hourly_test.ts, hr_measure)
    TnTx = DataFrames.combine(
        DataFrames.groupby(hourly_test, :ts_day), 
        df -> DataFrame(
            Tn=minimum(df.temp), 
            Tx=maximum(df.temp),
            Tn_time=df[argmin(df.temp),:ts],
            Tx_time=df[argmax(df.temp),:ts],
            times_p_day=DataFrames.nrow(df),
        )
    )
    return TnTx
end
function find_nearest(isdList::DataFrame, USAF::Int, WBAN::Int, k_nearest::Int)
    test_station = isdList[(isdList.USAF.==USAF) .& (isdList.WBAN.==WBAN), :]
    @assert nrow(test_station) == 1
    distances = sqrt.((test_station.X_PRJ.-isdList.X_PRJ).^2 
                   .+ (test_station.Y_PRJ.-isdList.Y_PRJ).^2)
    nearest_rows = sortperm(distances)
    nearest_and_test = isdList[nearest_rows[1:2+k_nearest-1], :]
    return nearest_and_test
end
