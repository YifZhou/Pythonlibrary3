def ackBar2(
        nTrap_s,
        eta_trap_s,
        tau_trap_s,
        nTrap_f,
        eta_trap_f,
        tau_trap_f,
        tExp,
        cRates,
        exptime=180,
        trap_pop_s=0,
        trap_pop_f=0,
        dTrap_f=[0],
        dTrap_s=[0],
        lost=0,
        mode='scanning'
):
    """Hubble Space Telescope ramp effet model

    Parameters:

    nTrap_s -- Number of slow traps in one pixel
    eta_trap_s -- Trapping efficiency for slow traps
    tau_trap_s -- Trap life time of slow trap
    nTrap_f -- Number of fast traps in one pixel
    eta_trap_f -- Trapping efficiency for fast traps
    tau_trap_f -- Trap life time of fast trap
    tExp -- start time of every exposures
    cRate -- intrinsic count rate of each exposures, unit e/s
    expTime -- (default 180 seconds) exposure time of the time series
    trap_pop -- (default 0) number of occupied traps at the beginning of the observations
    dTrap -- (default [0])number of extra trap added in the gap between two orbits
    lost -- (default 0, no lost) proportion of trapped electrons that are not eventually detected
    (mode) -- (default scanning, scanning or staring, or others), for scanning mode
      observation , the pixel no longer receive photons during the overhead
      time, in staring mode, the pixel keps receiving elctrons
    """
    dTrap_f = itertools.cycle(dTrap_f)
    dTrap_s = itertools.cycle(dTrap_s)
    obsCounts = np.zeros(len(tExp))
    nTrap_s = abs(nTrap_s)
    eta_trap_s = abs(eta_trap_s)
    tau_trap_s = abs(tau_trap_s)
    nTrap_f = abs(nTrap_f)
    eta_trap_f = abs(eta_trap_f)
    tau_trap_f = abs(tau_trap_f)
    trap_pop_s = min(trap_pop_s, nTrap_s)
    trap_pop_f = min(trap_pop_f, nTrap_f)
    for i in range(len(tExp)):
        try:
            dt = tExp[i+1] - tExp[i]
        except IndexError:
            dt = exptime
        f_i = cRates[i]
        c1_s = eta_trap_s * f_i / nTrap_s + 1 / tau_trap_s  # a key factor
        c1_f = eta_trap_f * f_i / nTrap_f + 1 / tau_trap_f
        # number of trapped electron during one exposure
        dE1_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * exptime))
        dE1_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * exptime))
        dE1_s = min(trap_pop_s + dE1_s, nTrap_s) - trap_pop_s
        dE1_f = min(trap_pop_f + dE1_f, nTrap_f) - trap_pop_f
        trap_pop_s = min(trap_pop_s + dE1_s, nTrap_s)
        trap_pop_f = min(trap_pop_f + dE1_f, nTrap_f)
        obsCounts[i] = f_i * exptime - dE1_s - dE1_f
        if dt < 5 * exptime:  # whether next exposure is in next batch of exposures
            # same orbits
            if mode == 'scanning':
                # scanning mode, no incoming flux between orbits
                dE2_s = - trap_pop_s * (1 - np.exp(-(dt - exptime)/tau_trap_s))
                dE2_f = - trap_pop_f * (1 - np.exp(-(dt - exptime)/tau_trap_f))
            elif mode == 'staring':
                # else there is incoming flux
                dE2_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * (dt - exptime)))
                dE2_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * (dt - exptime)))
            else:
                # others, same as scanning
                dE2_s = - trap_pop_s * (1 - np.exp(-(dt - exptime)/tau_trap_s))
                dE2_f = - trap_pop_f * (1 - np.exp(-(dt - exptime)/tau_trap_f))
            trap_pop_s = min(trap_pop_s + dE2_s, nTrap_s)
            trap_pop_f = min(trap_pop_f + dE2_f, nTrap_f)
        elif dt < 1200:
            if mode == 'staring':
                # next batch
                trap_pop_s = min(trap_pop_s * np.exp(-(dt-exptime)/tau_trap_s), nTrap_s)
                trap_pop_f = min(trap_pop_f * np.exp(-(dt-exptime)/tau_trap_f), nTrap_f)
            # else:
            #     trap_pop_s = min(trap_pop_s + dE2_s, nTrap_s)
            #     trap_pop_f = min(trap_pop_f + dE2_f, nTrap_f)
        else:
            trap_pop_s = min(trap_pop_s * np.exp(-(dt-exptime)/tau_trap_s) + next(dTrap_s), nTrap_s)
            trap_pop_f = min(trap_pop_f * np.exp(-(dt-exptime)/tau_trap_f) + next(dTrap_f), nTrap_f)
        trap_pop_s = max(trap_pop_s, 0)
        trap_pop_f = max(trap_pop_f, 0)

    return obsCounts
