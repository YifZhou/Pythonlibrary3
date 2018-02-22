import numpy as np
import pandas as pd
import os
from os import path
from lmfit import Parameters, Model
from misc import ackBar2
from misc import rebin
import shelve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle


def rampProfile(crate, slope, dTrap_s, dTrap_f, trap_pop_s, trap_pop_f, tExp,
                expTime):
    """Ramp profile for single directional scan

    And ackbar model parameters: number of traps, trapping coeeficient
    and trap life time

    :param crate: average count rate in electron/second
    :param slope: visit-long slope
    :param dTrap_s: extra trapped slow charges between orbits
    :param dTrap_f: extra trapped fast charges between orbits
    :param trap_pop_s: initially trapped slow charges
    :param trap_pop_f: initially trapped fast charges
    :param tExp: beginning of each exposure
    :param expTime: exposure time
    :returns: observed counts
    :rtype: numpy.array

    """

    tExp = (tExp - tExp[0])
    cRates = crate * (1 + tExp * slope / 1e7) / expTime
    obsCounts = ackBar2(
        cRates,
        tExp,
        expTime,
        trap_pop_s,
        trap_pop_f,
        dTrap_s=[dTrap_s],
        dTrap_f=[dTrap_f],
        dt0=[0],
        lost=0,
        mode='scanning')
    return obsCounts


def ackBarCorrector1(t, weights, counts, p, expTime, MCMC=False):
    """correct the ackbar model for one directional scan observations

    :param t: time stamps of the exposures
    :param orbits: orbit number of the exposures
    :param counts: observed counts
    :param p: Parameters objects to fit
    :param expTime: exposure time
    :returns: RECTE profile for correciting the light curve, best fit
    count rate array, ackbar output, slope
    :rtype: tuple of four numpy array

    """
    p = p.copy()
    p.add('crate', value=counts.mean(), vary=True)
    p.add('slope', value=0, min=-3, max=3, vary=True)
    rampModel = Model(rampProfile, independent_vars=['tExp', 'expTime'])
    t0 = t - t[0]  # make the first element in time array 0
    fitResult = rampModel.fit(
        counts,
        tExp=t0,
        expTime=expTime,
        params=p,
        weights=weights,
        method='nelder')
    fitResult = rampModel.fit(
        counts,
        tExp=t0,
        expTime=expTime,
        params=fitResult.params,
        weights=weights,
        method='powell')
    if MCMC:
        # TODO: add support for mcmc fit
        raise(NotImplementedError)
    print(fitResult.best_values)
    ackBar_in = fitResult.params['crate'].value * (
        1 + t0 * fitResult.params['slope'] / 1e7)
    ackBar_out = fitResult.best_fit
    ramp = ackBar_out / ackBar_in
    crates = fitResult.params['crate'].value * (1 + t0*fitResult.params['slope']/1e7)
    slope = (1 + t0*fitResult.params['slope']/1e7)
    return ramp, crates, slope


def rampProfile2(crate1, slope1, crate2, slope2, dTrap_s, dTrap_f, trap_pop_s,
                 trap_pop_f, tExp, expTime, scanDirect):
    """Ramp profile for bi-directional scan And ackbar model parameters:

    :param crate1: average count rate in electron/second for two
    directions
    :param slope1: visit-long slope for two directions
    :param crate2: average count rate in electron/second for two directions
    :param slope2: visit-long slope for two directions
    :param dTrap_s: extra trapped slow charges between orbits
    :param dTrap_f: extra trapped fast charges between orbits
    :param trap_pop_s: initially trapped slow charges
    :param trap_pop_f: initially trapped fast charges
    :param tExp: beginning of each exposure
    :param expTime: exposure time
    :param scanDirect: scan direction (0 or 1) for each exposure
    :returns: observed counts
    :rtype: numpy.array

    """
    tExp = (tExp - tExp[0])
    upIndex, = np.where(scanDirect == 0)
    downIndex, = np.where(scanDirect == 1)
    cRates = np.zeros_like(tExp, dtype=float)
    cRates[upIndex] = (crate1 * (1 + tExp * slope1 / 1e7) / expTime)[upIndex]
    cRates[downIndex] = (crate2 * (1 + tExp * slope2 / 1e7) / expTime)[downIndex]
    obsCounts = ackBar2(
        cRates,
        tExp,
        expTime,
        trap_pop_s,
        trap_pop_f,
        dTrap_f=[dTrap_f],
        dTrap_s=[dTrap_s],
        dt0=[0],
        lost=0,
        mode='scanning')
    return obsCounts


def ackBarCorrector2(t, weights, counts, p, expTime, scanDirect, MCMC=False):
    """correct the ackbar model for one directional scan observations

    :param t: time stamps of the exposures
    :param orbits: orbit number of the exposures
    :param counts: observed counts
    :param p: Parameters objects to fit
    :param expTime: exposure time
    :param scanDirect: scan direction (0 or 1) for each exposure
    :returns: RECTE profile for correciting the light curve, best fit
    count rate array, ackbar output, slope
    :rtype: tuple of four numpy array

    """
    upIndex, = np.where(scanDirect == 0)
    downIndex, = np.where(scanDirect == 1)
    p = p.copy()
    p.add('crate1', value=counts.mean(), vary=True)
    p.add('crate2', value=counts.mean(), vary=True)
    p.add('slope1', value=0, min=-5, max=5, vary=True)
    p.add('slope2', value=0, min=-5, max=5, vary=True)
    rampModel2 = Model(
        rampProfile2, independent_vars=['tExp', 'expTime', 'scanDirect'])
    # model fit, obtain crate, and transit parameter,
    # but ignore transit para for this time
    t0 = t - t[0]  # make the first element in time array 0
    fitResult = rampModel2.fit(
        counts,
        tExp=t0,
        expTime=expTime,
        scanDirect=scanDirect,
        weights=weights,
        params=p,
        method='nelder')
    fitResult = rampModel2.fit(
        counts,
        tExp=t0,
        expTime=expTime,
        scanDirect=scanDirect,
        weights=weights,
        params=fitResult.params,
        method='powell')
    if MCMC:
        # TODO: add support for mcmc fit
        raise(NotImplementedError)
    fitResult.params.pretty_print(columns=['value'])
    counts_fit = np.zeros_like(counts, dtype=float)
    counts_fit[upIndex] = (fitResult.params['crate1'].value * (
        1 + t0 * fitResult.params['slope1'] / 1e7))[upIndex]
    counts_fit[downIndex] = (fitResult.params['crate2'].value * (
        1 + t0 * fitResult.params['slope2'] / 1e7))[downIndex]
    ackBar_out = fitResult.best_fit
    ackBar_in = np.zeros_like(ackBar_out)
    ackBar_in[upIndex] = fitResult.params['crate1'].value * (
        1 + t0[upIndex] * fitResult.params['slope1'] / 1e7)
    ackBar_in[downIndex] = fitResult.params['crate2'].value * (
        1 + t0[downIndex] * fitResult.params['slope2'] / 1e7)
    ramp = ackBar_out / ackBar_in
    slope = np.zeros_like(ackBar_out)
    slope[upIndex] = 1 + t0[upIndex] * fitResult.params['slope1'] / 1e7
    slope[downIndex] = 1 + t0[downIndex] * fitResult.params['slope2'] / 1e7
    crates = np.zeros_like(ackBar_out)
    crates[upIndex] = fitResult.params['crate1'] * slope[upIndex]
    crates[downIndex] = fitResult.params['crate2'] * slope[downIndex]
    return ramp, crates, slope


def visit_deRamp(pDeRamp,
                 time,
                 LCmatrix,
                 Errmatrix,
                 weights,
                 expTime,
                 twoDirect=False,
                 scanDirect=None,
                 plot=False,
                 plotDIR='.',
                 MCMC=False):
    """
    fit transit models
    deAckbar for a visit
    """
    nLC = LCmatrix.shape[0]  # number of light curves
    rampMat = LCmatrix.copy()
    slopeMat = LCmatrix.copy()
    p = pDeRamp.copy()
    print("Ramp fitting start. {0} channels to be corrected".format(nLC))
    if plot and (not path.exists(plotDIR)):
        os.makedirs(plotDIR)
    for i in range(nLC):
        print("fitting ramps for channel {0:02d}/{1}".format(i, nLC))
        if twoDirect:
            ramp, crates, slope = ackBarCorrector2(
                time, weights, LCmatrix[i, :], p, expTime, scanDirect, MCMC)
        else:
            ramp, crates, slope = ackBarCorrector1(
                time, weights, LCmatrix[i, :], p, expTime, MCMC)
        rampMat[i, :] = ramp
        slopeMat[i, :] = slope
        if plot:
            plt.close('all')
            print("making ramp fit plot for channel {0:02d}/{1}".format(i, nLC))
            fig, ax = plt.subplots()
            ax.errorbar(time / 3600, LCmatrix[i, :], yerr=Errmatrix[i, :], ls='none')
            ax.plot(time / 3600, ramp * crates)
            ax.set_xlabel('Time [hour]')
            ax.set_ylabel('Count [e$^-$]')
            ax.set_title('RECTE Ramp Fit for Channel {0:02d}/{1}'.format(i, nLC))
            saveFN = path.join(plotDIR, 'Ramp_fit_Cannel_{0:02d}.pdf'.format(i))
            plt.savefig(saveFN)
            fig, ax = plt.subplots()
            ax.errorbar(time / 3600, LCmatrix[i, :] / (ramp * crates),
                        yerr=Errmatrix[i, :] / (ramp * crates), ls='none')
            ax.set_xlabel('Time [hour]')
            ax.set_ylabel('Relative flux')
            ax.set_title('RECTE Ramp Removed for Channel {0:02d}/{1}'.format(i, nLC))
            saveFN = path.join(plotDIR, 'Ramp_removed_Cannel_{0:02d}.pdf'.format(i))
            plt.savefig(saveFN)
    return rampMat, slopeMat


def scanDeRamp(p0,
               weights,
               expTime,
               LCmatrix,
               Errmatrix,
               time,
               xList,
               twoDirect=False,
               scanDirect=None,
               binSize=10,
               plot=False,
               plotDIR='.',
               MCMC=False):
    # properly bin the
    xBinLEdge = np.arange(xList.min(), xList.max(), binSize)  # left edge of each bin
    xBinIndex = np.digitize(xList, xBinLEdge) - 1
    mat_binned = np.array([
        rebin(LCmatrix[:, j], binSize, method='mean')
        for j in range(len(time))
    ]).T
    err_binned = np.array([
        rebin(Errmatrix[:, j], binSize, method='meansq') / np.sqrt(binSize)
        for j in range(len(time))
    ]).T
    rampMat, slopeMat = visit_deRamp(
        p0, time, mat_binned, err_binned, weights, expTime, twoDirect,
        scanDirect, plot, plotDIR, MCMC)
    LCmatrix_deramp = LCmatrix.copy()
    Errmatrix_deramp = Errmatrix.copy()
    correctionMatrix = Errmatrix.copy()
    for i in range(len(xList)):
        ramp_i = rampMat[xBinIndex[i], :]
        slope_i = rampMat[xBinIndex[i], :]
        LCmatrix_deramp[i, :] = LCmatrix[i, :] / ramp_i / slope_i
        correctionMatrix[i, :] = ramp_i * slope_i
    return LCmatrix_deramp, Errmatrix_deramp, correctionMatrix


if __name__ == '__main__':
    pass
    # nTrap_s = 1525.38  # 1320.0
    # eta_trap_s = 0.013318  # 0.01311
    # tau_trap_s = 1.63e4
    # nTrap_f = 162.38
    # eta_trap_f = 0.008407
    # tau_trap_f = 281.463
    # p = Parameters()
    # p.add('trap_pop_s', value=0, min=0, max=nTrap_s, vary=True)
    # p.add('trap_pop_f', value=0, min=0, max=nTrap_f, vary=True)
    # p.add('dTrap_f', value=0, min=0, max=nTrap_f, vary=True)
    # p.add('dTrap_s', value=50, min=0, max=nTrap_s, vary=True)
    # main(p, 1, 'Trappist_1_Visit_1')
