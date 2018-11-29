#!/usr/bin/env python
"""in HST fits header time is given in MJD UTC
in transit, normally BJD_TDB is used
This function convert MJD_UTC to BJD_TDB
"""
from astropy.time import Time


def JD2BJD(jd, target_coord, site_location=('-0.0077d', '51.4826')):
    """do the conversion
    mjd -- input mjd, scale is utc
    target_coord -- the coordinate of the target, in astropy.coord format,
    to caculate the light travel time
    site_location -- location of the telescope, to make accurate calcualtion of
    light travel time and tdb conversion. Not very important here. The default value
    is for Greenwich Observatory.
"""
    t = Time(jd, format='jd', scale='utc', location=site_location)
    # calculate light travel time
    ltt = t.light_travel_time(target_coord)
    # print(t, ltt)
    # convert t to tdb, and add light travel time
    t_out = (t.tdb + ltt).jd
    return t_out
