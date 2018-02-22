#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from .HSTtiming import HSTtiming


def HSTsinLC(expTime, cRate, param,
             orbits=4,
             orbitLength=96,  # min
             visibility=50,  # min
             overhead=20,  # s
             plot=False):
    """Simulate a sinusoidal signal observed by HST

    :param expTime: exposure time
    :param cRate: average count rate
    :param param: parameters describing the sinusoid, a dictionary
    :param orbits: number of orbits
    :param orbit: (default 96 min) length of a orbit
    :param visibility: (default 50 min) length of visible period per orbit
    :param overhead: overhead [s] per exposure
    :param plot: wheter to make a plot
    :returns: count, t
    :rtype: tuple

    """

    t = HSTtiming(expTime, orbits, orbitLength, visibility, overhead)
    # calculate count, be careful that period in h
    count = cRate * expTime * \
        (1 + np.sin((2 * np.pi * t / (param['period'] * 3600)) +
                    param['phase']) * param['amplitude'])

    if plot:
        fig, ax = plt.subplots()
        ax.plot(t/60, count, 'o')
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('count [$\mathsf{e^-}$]')
        plt.show()
    return count, t


if __name__ == '__main__':
    plt.close('all')
    param = {}
    param['period'] = 14.3
    param['phase'] = np.random.uniform(0, 2*np.pi, 1)
    param['amplitude'] = 0.05
    count, t = HSTsinLC(88.9, 250, param,
                        orbits=6, plot=True)
    plt.show()
