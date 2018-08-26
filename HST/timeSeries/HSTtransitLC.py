#!/usr/bin/env python
"""simulate a sin wave observed by HST
default orbit length 96 min,
default visibility per orbit, 50 min
defaut overhead 20s
return the electron count for each exposure
"""


import matplotlib.pyplot as plt
import batman
import numpy as np
from HST.timeSeries.HSTtiming import HSTTiming


def HSTtransitLC(expTime, cRate,
                 param,
                 orbits=4,
                 orbitLength=96,  # min
                 visibility=50,  # min
                 overhead=20,  # s
                 plot=False):
    """Simulate a transit signal observed by HST

    :param expTime: exposure time
    :param cRate: average count rate
    :param param: parameters describing the sinusoid, a batman dictionary
    :param orbits: number of orbits
    :param orbit: (default 96 min) length of a orbit
    :param visibility: (default 50 min) length of visible period per orbit
    :param overhead: overhead [s] per exposure
    :param plot: wheter to make a plot
    :returns: count, t
    :rtype: tuple

    """
    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = param['t0'] / (24 * 60)  # time of inferior conjunction
    params.per = param['period']  # orbital period
    params.rp = param['rp']  # planet radius (in units of stellar radii)
    params.a = param['a']  # semi-major axis (in units of stellar radii)
    params.ecc = 0  # eccentricity
    params.inc = 89.1  # inclination
    params.w = 90.  # longitude of periastron (in degrees)
    params.limb_dark = "linear"  # limb darkening model
    params.u = [0.28]  # limb darkening coefficients

    t = HSTTiming(expTime, orbits, orbitLength, visibility, overhead)
    m = batman.TransitModel(params, t / (24 * 3600))
    # calculate count, be careful that period in h
    count = cRate * expTime * m.light_curve(params)
    t_mod = np.linspace(t.min(), t.max(), 10*len(t))
    m = batman.TransitModel(params, t_mod / (24*3600))
    lc_mod = m.light_curve(params)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(t / 60, count, 'o')
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('count [$\mathsf{e^-}$]')
        plt.show()
    return count, t, lc_mod, t_mod


if __name__ == '__main__':
    plt.close('all')
    # for GJ 1214b # from Kreidberg 2013
    params = {}
    params['period'] = 1.58040464894  # days
    params['rp'] = 0.0134**0.5  # stellar radii
    params['t0'] = 220
    count, t = HSTtransitLC(88.9, 250, params, orbits=4, plot=True, overhead=50)
    plt.show()
