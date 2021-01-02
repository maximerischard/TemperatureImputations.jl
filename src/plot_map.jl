using PyCall
using Printf
@pyimport mpl_toolkits.basemap as basemap
cbbPalette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

using PyCall
@pyimport cartopy
@pyimport cartopy.crs as ccrs
@pyimport cartopy.feature as cfeature
@pyimport cartopy.io.img_tiles as cimgt
@pyimport matplotlib

function plot_map(
        isdSubset, epsg::Int;
        first_test::Bool=false,
        horizontalalignment="auto",
        zoomlevel=8,
        tickspace_km=50.0, # km
        )
    crs = cartopy.crs.epsg(epsg)
    
    # Create a Stamen terrain background instance.
    stamen_terrain = cimgt.Stamen("terrain-background")
    
    # Create a GeoAxes in the tile's projection.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=crs)

    # Limit the extent of the map to a small longitude/latitude range.
    xmin = floor(minimum(isdSubset.X_PRJ); digits=-5) # to the nearest 100km
    xmax = ceil( maximum(isdSubset.X_PRJ)+20e3; digits=-5) # leave room for text labels
    ymin = floor(minimum(isdSubset.Y_PRJ); digits=-5)
    ymax = ceil( maximum(isdSubset.Y_PRJ); digits=-5)
    ax.set_extent([xmin-10e3, xmax+10e3, ymin-10e3, ymax+10e3], crs=crs)
    
    # Add the Stamen data
    ax.add_image(stamen_terrain, zoomlevel)
    
    # Add State lines
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "10m",
        edgecolor="black", facecolor="none"))

    # Tick marks and axis labels
    fmt_km = matplotlib.ticker.FuncFormatter(py"""lambda x, p: format(x/1000, ",.0f")"""o)
    ax.set_xticks(range(xmin; stop=xmax, step=tickspace_km*1e3))
    ax.set_yticks(range(ymin; stop=ymax, step=tickspace_km*1e3))
    ax.xaxis.set_major_formatter(fmt_km)
    ax.yaxis.set_major_formatter(fmt_km)
    plt.xlabel("Eastings (km)")
    plt.ylabel("Northings (km)")
    
    # Add crosses for each weather station
    for i in 1:nrow(isdSubset)
        x, y = isdSubset[i, :X_PRJ], isdSubset[i, :Y_PRJ]
        if !ismissing(isdSubset[i, :ICAO])
            label = isdSubset[i, :ICAO]
        else
            label = @sprintf("%d", isdSubset[i, :USAF])
        end
        if i == 1 && first_test
            plt.plot(x, y, "o", color=cbbPalette[6], zorder=4)
            plt.plot(x, y, "o", color=cbbPalette[6],
                     transform=crs)
        else
            plt.plot(x, y, "x", color="black",
                     transform=crs)
        end
        if x > (xmin+xmax)/2
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
    plt.text(0.99, 0.01, "Map tiles by Stamen Design, under CC BY 3.0. Data © OpenStreetMap contributors.",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=5,
        bbox=Dict(:edgecolor=>"none",:facecolor=>"white", :alpha=>0.7, :pad=>1.0),
        transform = plt.gca()[:transAxes],)
end
