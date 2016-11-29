# ! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import itertools
"""ramp effect model

author: Daniel Apai

Version 0.2: add extra keyword parameter to indicate scan or staring
mode observations for staring mode, the detector receive flux in the
same rate during overhead time as that during exposure
precise mathematics forms are included

Version 0.1: Adapted original IDL code to python by Yifan Zhou

"""


def ackBar(
        nTrap,
        eta_trap,
        tau_trap,
        tExp,
        cRates,
        exptime=180,
        trap_pop=0,
        dTrap=[0],
        lost=0,
        mode='scanning'
):
    """Hubble Space Telescope ramp effet model

    Parameters:

    nTrap -- Number of traps in one pixel
    eta_trap -- Trapping efficiency
    tau_trap -- Trap life time
    tExp -- start time of every exposures
    cRate -- intrinsic count rate of each exposures
    expTime -- (default 180 seconds) exposure time of the time series
    trap_pop -- (default 0) number of occupied traps at the beginning of the observations
    dTrap -- (default [0])number of extra trap added in the gap between two orbits
    lost -- (default 0, no lost) proportion of trapped electrons that are not eventually detected
    (mode) -- (default scanning, scanning or staring, or others), for scanning mode
      observation , the pixel no longer receive photons during the overhead
      time, in staring mode, the pixel keps receiving elctrons
    """
    dTrap = itertools.cycle(dTrap)
    obsCounts = np.zeros(len(tExp))
    nTrap = abs(nTrap)
    eta_trap = abs(eta_trap)
    tau_trap = abs(tau_trap)
    for i in range(len(tExp)):
        try:
            dt = tExp[i+1] - tExp[i]
        except IndexError:
            dt = exptime
        f_i = cRates[i]
        c1 = eta_trap * f_i / nTrap + 1 / tau_trap  # a key factor
        # number of trapped electron during one exposure
        dE1 = (eta_trap * f_i / c1 - trap_pop) * (1 - np.exp(-c1 * exptime))
        trap_pop = trap_pop + dE1
        obsCounts[i] = f_i * exptime - dE1
        if dt < 5 * exptime:  # whether next exposure is in next batch of exposures
            # same orbits
            if mode == 'scanning':
                # scanning mode, no incoming flux between orbits
                dE2 = - trap_pop * (1 - np.exp(-(dt - exptime)/tau_trap))
            elif mode == 'staring':
                # else there is incoming flux
                dE2 = (eta_trap * f_i / c1 - trap_pop) * (1 - np.exp(-c1 * (dt - exptime)))
            else:
                # others, same as scanning
                dE2 = - trap_pop * (1 - np.exp(-(dt - exptime)/tau_trap))
            trap_pop = min(trap_pop + dE2, nTrap)
        elif dt < 1200:
            # next exposure gap
            trap_pop = min(trap_pop * np.exp(-(dt-exptime)/tau_trap), nTrap)
        else:
            # next orbits
            trap_pop = min(trap_pop * np.exp(-(dt-exptime)/tau_trap) + next(dTrap), nTrap)
        trap_pop = max(trap_pop, 0)
        # out_trap = max(-(trap_pop * (1 - np.exp(exptime / tau_trap))), 0)
        # out_trap = trap_pop / tau_trap * dt * np.exp(-dt / tau_trap)

    return obsCounts


if __name__ == '__main__':
    t1 = np.linspace(0, 2700, 80)
    t2 = np.linspace(5558, 8280, 80)
    t = np.concatenate((t1, t2))
    crate = 100
    crates = crate * np.ones(len(t))
    dataDIR = '/Users/ZhouYf/Documents/HST14241/alldata/2M0335/DATA/'
    from os import path
    import pandas as pd

    info = pd.read_csv(
        path.expanduser('~/Documents/HST14241/alldata/2M0335/2M0335_fileInfo.csv'),
        parse_dates=True,
        index_col='Datetime')
    info['Time'] = np.float32(info.index - info.index.values[0]) / 1e9
    expTime = info['Exp Time'].values[0]
    grismInfo = info[info['Filter'] == 'G141']
    tExp = grismInfo['Time'].values
    # cRates = np.ones(len(LC)) * LC.mean() * 1.002
    cRates = np.ones(len(tExp)) * 80
    obs = ackBar(1200, 0.02, 5000, tExp, cRates, exptime=expTime, lost=0,
                 dTrap=[200], mode='scanning')
    obs2 = ackBar(1200, 0.02, 5000, tExp, cRates, exptime=expTime, lost=0,
                 dTrap=[200], mode='staring')
    plt.close('all')
    # plt.plot(tExp, LC*expTime, 'o')
    plt.plot(tExp, obs, '-')
    plt.plot(tExp, obs2, '-')
    # plt.ylim([crate * 0.95, crate * 1.02])
    plt.show()
