doc = """

    Usage:
        test_stations.jl choose <data_dir> <outputfile> --epsg=<epsg> --maxgap=<maxgap_hours>
        test_stations.jl tojson <csvfile> <outputfile>
"""
using DocOpt: docopt

import Shapefile
import Proj4
import CSV, JSON
using DataFrames
using Dates: Hour, Millisecond

import TemperatureImputations

using LibGEOS: centroid, envelope, MultiPolygon, Point, within
using GeoInterface: coordinates


function main()
    arguments = docopt(doc)
    @show arguments
    outputfile = arguments["<outputfile>"]

    if arguments["choose"]
        epsg = parse(Int, arguments["--epsg"])
        maxgap_hours = parse(Float64, arguments["--maxgap"])
        data_dir = arguments["<data_dir>"]

        isdList = TemperatureImputations.read_isdList(; data_dir=data_dir, epsg=epsg)
        isd_wData = TemperatureImputations.stations_with_data(isdList; data_dir=data_dir)

        isd_smallgap = filter_stations_with_small_gaps(isd_wData, data_dir, maxgap_hours)

        states_shapefile_path = joinpath(data_dir,"cb_2017_us_state_5m","cb_2017_us_state_5m.shp")
        states_shapefile = open(states_shapefile_path, "r") do io
            read(io, Shapefile.Handle)
        end
        inearcentroids = get_stations_near_centroids(isd_smallgap, states_shapefile, epsg)
        test_stations = isd_smallgap[inearcentroids,:]
        CSV.write(joinpath(outputfile), test_stations)
        println("Test stations written to: ", outputfile)
    elseif arguments["tojson"]
        csvfile = arguments["<csvfile>"]
        test_stations = CSV.read(csvfile, DataFrame)
        json_dict = Dict("ICAO" => test_stations.ICAO)
        open(joinpath(outputfile), "w") do io
            JSON.print(io, json_dict)
        end
    end
end

function filter_stations_with_small_gaps(isd_stations, data_dir::String, maxgap::Float64)
    gaps = Float64[]
    for istation in 1:nrow(isd_stations)
        usaf, wban, icao = isd_stations[istation, :USAF], isd_stations[istation, :WBAN], isd_stations[istation, :ICAO]
        hourly = TemperatureImputations.read_station(usaf, wban, 42; data_dir=data_dir);
        ts = hourly.ts
        ms_per_hour = convert(Millisecond, Hour(1))
        max_gap = maximum(diff(ts)) / ms_per_hour
        push!(gaps, max_gap)
    end

    # stations without a gap in data greater than <maxgap> hours:
    isd_smallgap = isd_stations[gaps .< maxgap, :]
    return isd_smallgap
end

function get_stations_near_centroids(isd_stations, states_shapefile, epsg::Int)
    X_PRJ, Y_PRJ = isd_stations.X_PRJ, isd_stations.Y_PRJ
    LAT, LON = isd_stations.LAT, isd_stations.LON
    station_points = Point.(LON, LAT)

    wgs84 = Proj4.Projection("+proj=longlat +datum=WGS84 +no_defs")
    epsg_crs = Proj4.Projection(Proj4.epsg[epsg])

    inearcentroids = Int64[]
    for stateshape in states_shapefile.shapes
        poly = MultiPolygon(coordinates(stateshape))
        centroid_lonlat = coordinates(centroid(poly))
        centroid_proj = Proj4.transform(wgs84, epsg_crs, centroid_lonlat)
        # restrict to airports within state
        in_state = findall(within.(station_points, Ref(poly)))
        if length(in_state)==0
            continue
        end
        distances = euclidean_distance.(X_PRJ[in_state], centroid_proj[1],
                                        Y_PRJ[in_state], centroid_proj[2])
        inearest = in_state[argmin(distances)]
        push!(inearcentroids, inearest)
    end
    return inearcentroids
end
euclidean_distance(x1, y1, x2, y2) = sqrt((x1-x2)^2+(y1-y2)^2)

main()
