struct FittingWindow{D<:Union{Date,DateTime}}
    start_date::D
    end_date::D
end
function add_time_to_window(fw::FittingWindow{Date}, hr_measure::Hour)
    FittingWindow(
        DateTime(fw.start_date) - Day(1) + hr_measure,
        DateTime(fw.end_date)            + hr_measure
    )
end

function predictions_dirpath(;save_dir::String, crossval::Bool, GPmodel::String, icao::String)
    joinpath(save_dir, "predictions_from_nearby", crossval ? "crossval" : "mll", GPmodel, icao)
end
function predictions_fname(;usaf::Int, wban::Int, icao::String, fw::FittingWindow)
     @sprintf("%d_%d_%s_%s_to_%s.jld", 
        usaf, wban, icao,
        Date(fw.start_date), Date(fw.end_date))
end
function load_predictions(nearby_dir::String, usaf::Int, wban::Int, icao::String, fw::FittingWindow)
    nearby_fname = predictions_fname(;usaf=usaf, wban=wban, icao=icao, fw=fw)
    nearby_path = joinpath(nearby_dir, nearby_fname)
    return load(nearby_path)["nearby_pred"]
end
function load_predictions(;save_dir::String, crossval::Bool, GPmodel::String, icao::String,
                           usaf::Int, wban::Int, fw::FittingWindow)
    nearby_dir = predictions_dirpath(;save_dir=save_dir, crossval=crossval, GPmodel=GPmodel, icao=icao)
    return load_predictions(nearby_dir, usaf, wban, icao, fw)
end

# function stan_dirpath(
        # ;save_dir::String, crossval::Bool, GPmodel::String,
         # hr_measure::Hour, usaf::Int, wban::Int, icao::String, fw::FittingWindow)
    # return joinpath(
            # save_dir,
            # "hr_measure",
            # crossval ? "crossval" : "mll",
            # GPmodel,
            # string(Int(hr_measure.value)), 
            # icao,
            # @sprintf("%d_%d_%s_%s_to_%s/", 
                     # usaf, wban, icao, Date(fw.start_date), Date(fw.end_date)
                     # )
    # ) |> abspath
# end


function get_nearby(fw::FittingWindow, GPmodel::AbstractString, usaf::Int, wban::Int, icao::String, saved_dir::String; crossval::Bool)
    pred_dir = joinpath(saved_dir, "predictions_from_nearby", crossval ? "crossval" : "mll", GPmodel, icao)
    pred_fname = predictions_fname(usaf, wban, icao, fw)
    pred_fpath = joinpath(pred_dir, pred_fname)
    nearby_pred = load(pred_fpath)["nearby_pred"]
    return nearby_pred
end

function predict_from_nearby_chunks()
    fitting_windows = FittingWindow{DateTime}[]

    dt_start=DateTime(2015,1,1,0,0,0)
    mintime = DateTime(2015,1,1,0,0,0)
    maxtime = DateTime(2016,1,1,0,0,0)
    increm=(maxtime-mintime) / 15
    window=3*increm
    dt_start = mintime

    while true
        dt_end=dt_start+window
        fw = FittingWindow(dt_start, dt_end)
        push!(fitting_windows, fw)
        if dt_end >= maxtime
            break
        end
        dt_start+=increm
    end
    return fitting_windows
end

function imputation_chunks(; stan_days::Day)
    stan_windows = FittingWindow{Date}[]

    stan_increment = stan_days - Day(4)
    firstday = Date(2015,1,1)
    lastday = Date(2015,12,31)
    windownum = 1
    while true
        stan_start = firstday + (windownum-1)*stan_increment
        stan_end = stan_start + stan_days - Day(1)
        sw = FittingWindow(stan_start, stan_end)
        push!(stan_windows, sw)
        if stan_end >= lastday
            break
        end
        windownum += 1
    end
    return stan_windows
end
function overlap(a::FittingWindow, b::FittingWindow)
    # conditions that imply the windows don't overlap at all:
    a_after_b = a.start_date >= b.end_date
    b_after_a = b.start_date >= a.end_date
    return !(a_after_b || b_after_a)
end
"""
    How much buffer time is there on either side of the window?
"""
function buffer(a::FittingWindow, b::FittingWindow)
    start_diff = a.start_date - b.start_date
    end_diff = b.end_date - a.end_date
    return min(start_diff, end_diff) # worst of the two
end
""" 
    Amongst a list of candidate windows `cand`, find the window that includes `wind`
    with the largest buffer on either sides.
"""
function find_best_window(wind::FittingWindow, cands::Vector{<:FittingWindow})
    incl_wdows = [fw for fw in cands if overlap(wind, fw)]
    buffers = [buffer(wind, fw) for fw in incl_wdows]
    imax = argmax(buffers) # maximum of minimum
    best_window = incl_wdows[imax]
    return best_window
end

"""
    How much buffer time is there on either side of the window?
"""
function buffer(t::DateTime, wt::FittingWindow)
    start_diff = t - wt.start_time
    end_diff = wt.end_time - t
    return min(start_diff, end_diff)
end

""" 
    Amongst a list of candidate windows `cand`, find the window that includes time `t`
    with the largest buffer on either sides.
"""
function find_best_window(t::DateTime, cands::Vector{FittingWindow{DateTime}})
    inside_windows = t_inside_wt.(t, cands)
    incl_wdows = cands[inside_windows]
    buffers = [buffer(t, wt) for wt in incl_wdows]
    imax = argmax(buffers)
    return find(inside_windows)[imax]
end

function t_inside_fw(t, wt::FittingWindow)
    return wt.start_time <= t <= wt.end_time
end
function t_inside_fw(t::DateTime, fw::FittingWindow{Date}, hr_measure::Hour)
    measure_day = TemperatureImputations.measurement_date(t, hr_measure)
    in_window = fw.start_date <= measure_day <= fw.end_date-Day(1)
    return in_window
end
