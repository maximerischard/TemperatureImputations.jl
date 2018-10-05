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

"""
    Distance on units sphere, 
    see (http://www.johndcook.com/blog/python_longitude_latitude/).
"""
function distance_on_unit_sphere(lat1, long1, lat2, long2)
 
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = π/180.0
         
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
         
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
         
    # Compute spherical distance from spherical coordinates.
         
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
     
    cosangle = (sin(phi1)*sin(phi2)*cos(theta1 - theta2) +
           cos(phi1)*cos(phi2))
    arc = acos( cosangle )
 
    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return arc
end
