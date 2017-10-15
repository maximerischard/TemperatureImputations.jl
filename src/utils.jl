using Base.Dates: Day, Hour

function old_measurement_date(t::DateTime, hr_measure::Hour)
    if Hour(t) <= hr_measure
        # if the time now is before the measurement time,
        # this temperature is part of today's record
        return Date(t)
    else
        # otherwise we'll have to wait until tomorrow
        return Date(t)+Day(1)
    end
end
function measurement_date(t::DateTime, hr_measure::Hour)
    day = trunc(t, Day)
    time = t - day
    if time <= hr_measure
        # if the time now is before the measurement time,
        # this temperature is part of today's record
        return Date(t)
    else
        # otherwise we'll have to wait until tomorrow
        return Date(t)+Day(1)
    end
end
