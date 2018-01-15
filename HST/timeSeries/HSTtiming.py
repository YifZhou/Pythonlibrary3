#!/usr/bin/env python
import numpy as np


def HSTTiming(exptime,
         orbits=4,
         orbitLength=96,  # min
         visibility=50,  # min
         overhead=20):
    """generate a series of time for the starting of exposure

    :param exptime: exposure time
    :param orbits: number of orbits
    :param orbitLength: (default 96 minutes) the length of one orbit
    :param visibility: (default 20 minutes) the length of visible time
    period per orbit
    :param overhead: (default 20  seconds) the length of overhead per exposure

    """
    t = np.arange(0, visibility * 60, (exptime + overhead))
    t0 = np.arange(orbits) * orbitLength * 60  # starting time of each orbits
    return np.concatenate([t + t0i for t0i in t0])
