using PyCall
using Printf
@pyimport mpl_toolkits.basemap as basemap
cbbPalette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

function plot_map(isdSubset, epsg::Int; first_test::Bool=false, arcgis::Bool=false, resolution="l", horizontalalignment="auto")
    minlon = minimum(isdSubset[:LON])
    maxlon = maximum(isdSubset[:LON])
    minlat = minimum(isdSubset[:LAT])
    maxlat = maximum(isdSubset[:LAT])
    
    # add a bit of padding
    pad = 0.1
    llcrnrlon = minlon - pad*(maxlon-minlon)
    llcrnrlat = minlat - pad*(maxlat-minlat)
    urcrnrlon = maxlon + pad*(maxlon-minlon)
    urcrnrlat = maxlat + pad*(maxlat-minlat)
    @show llcrnrlon, urcrnrlon
    @show llcrnrlat, urcrnrlat
    @show resolution

    map = basemap.Basemap(epsg=epsg, 
                          llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, 
                          resolution=resolution, suppress_ticks=false)
    if arcgis
        map[:arcgisimage](service="World_Shaded_Relief", xpixels = 1500, verbose=true, zorder=1, dpi=100)
    end
    map[:drawstates](linewidth=1.0, zorder=2, color=cbbPalette[1])
    
    llcrnrx, urcrnrx = map[:llcrnrx], map[:urcrnrx]
    width = urcrnrx - llcrnrx
    @show llcrnrx, urcrnrx

    x_0 = map[:projparams]["x_0"]
    y_0 = map[:projparams]["y_0"]
    @show x_0, y_0
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    
    for i in 1:nrow(isdSubset)
        lat, lon = isdSubset[i, :LAT], isdSubset[i, :LON]

        x, y = map(lon, lat)
        if !ismissing(isdSubset[i, :ICAO])
            label = isdSubset[i, :ICAO]
        else
            label = @sprintf("%d", isdSubset[i, :USAF])
        end
        if i == 1 && first_test
            plt.plot(x, y, "o", color=cbbPalette[6], zorder=4)
        else
            plt.plot(x, y, "x", color="black", zorder=4)
        end
        if lon > (llcrnrlon+urcrnrlon)/2
            # align right
            plt.annotate(label, xy=(x, y),  xycoords="data",
                            xytext=(0, 5), textcoords="offset points",
                            fontsize=8,
                            color="black",
                            horizontalalignment = horizontalalignment == "auto" ? "right" : horizontalalignment,
                            zorder=3
                            )
        else
            plt.annotate(label, xy=(x, y),  xycoords="data",
                            xytext=(0, 5), textcoords="offset points",
                            fontsize=8,
                            color="black",
                            horizontalalignment = horizontalalignment == "auto" ? "left" : horizontalalignment,
                            zorder=3
                            )
        end
    end

    # assumes EPSG unit is meters
    _xticks = plt.xticks()[1]
    plt.xticks(_xticks, [@sprintf("%.0f", x/1000) for x in _xticks])
    _yticks = plt.yticks()[1]
    plt.yticks(_yticks, [@sprintf("%.0f", x/1000) for x in _yticks])
    plt.xlabel("Eastings (km)")
    plt.ylabel("Northings (km)")
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.text(0.99, 0.01, "Esri, HERE, Garmin, FAO, NOAA | Copyright: 2014 Esri",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=5,
        bbox=Dict(:edgecolor=>"none",:facecolor=>"white", :alpha=>0.7, :pad=>1.0),
        transform = plt.gca()[:transAxes],)
end
;
