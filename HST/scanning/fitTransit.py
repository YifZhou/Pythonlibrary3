"""a template for transit profile fit In the transit, the model light
curve is a single transit light curve. With semi-major axis equals to
10 and inclination equals to 90. Quadratic limb darkening law is assumed

To apply the model to specific case, First make sure input parameter
.ini file is available and all the parameters are set correctly.  The
fitting parameters for broad and spectral bands should be provided, as
well as their min and max limits.  Parameter that are not involved in
the fit should be set in others section.


Then modify the transitModel function, transitModel_spectral and
transitModel_white if necessary. Also, the limb_darkening profile
should be confirmed. All functions/parameters that requires
modifications are hilighted with ``TODO" flags
"""

from misc import rebin
import numpy as np
import pandas as pd
import shelve
import pickle
from os import path
import batman
import emcee
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
# from scipy.stats import shapiro, kstest, probplot
from misc import makedirs, mcmcRunner
from plot import plotMCMC
import sys
from scipy.interpolate import interp1d
import configparser
import json
from HST.scanning import scanData
plt.style.use('paper')

# define the transit model as a global parameter
transit_params = batman.TransitParams()  # object to store transit parameters
transit_params.t0 = 0  # time of inferior conjunction
transit_params.per = 1.0  # orbital period
transit_params.rp = 0.1  # planet radius (in units of stellar radii)
transit_params.a = 10  # semi-major axis (in units of stellar radii)
transit_params.ecc = 0  # eccentricity
transit_params.inc = 90  # inclination
transit_params.w = 90.  # longitude of periastron (in degrees)
transit_params.limb_dark = "quadratic"  # TODO limb darkening model
transit_params.u = [0.15, 0.45]  # TODO limb darkening coefficients
# initialize the model
m = batman.TransitModel(transit_params, np.linspace(0, 10))


def lnprior(params, params_min, params_max):
    """TODO Generic flat prior distribution functions return -inf if params are
    outside of the parameter space defined by params_min and
    params_max. Otherwise, return 0

    :param params: parameter values
    :param params_min: upper
    :param params_max:
    :returns: ln of the prior probability
    :rtype: float
    """
    inSpace = 1
    for i, p in enumerate(params):
        inSpace = inSpace * (params_max[i] > p > params_min[i])
    if inSpace == 0:
        return -np.inf
    else:
        return 0


def transitModel_spectral(params, t, expTime, t01, t02):
    """TODO
    a generic transit light curve with batman

    :param params: transit planet parameters
    :param t: time stamps in seconds
    :param expTime: exposure time for each frame
    :returns: light curve model
    :rtype: np.array

    """
    transit_params.per = 1.51087081  # orbital period
    transit_params.a = 20.4209
    transit_params.inc = 89.65
    transit_params.t0 = t01  # time of inferior conjunction
    transit_params.rp = params[0]  # planet radius (in units of stellar radii)
    transit_params.u = [params[2],
                        params[3]]  # linear limb darkening coefficients
    m_b = batman.TransitModel(
        transit_params, (t - 0.5 * expTime) / 86400, exp_time=expTime / 86400)
    lc1 = m_b.light_curve(transit_params)

    transit_params.per = 2.4218233  # orbital period
    transit_params.a = 27.9569
    transit_params.inc = 89.67
    transit_params.t0 = t02
    transit_params.rp = params[1]  # planet radius (in units of stellar radii)
    transit_params.u = [params[2], params[3]]  # linear limb darkening

    m_c = batman.TransitModel(
        transit_params, (t - 0.5 * expTime) / 86400, exp_time=expTime / 86400)
    lc2 = m_c.light_curve(transit_params)
    lc = (lc1 + lc2) - 1  # two transit, remove one baseline
    return lc


def lnlike_spectral(params, t, f, ferr, expTime, t01, t02):
    """transit profile likelihood function

    :param params: transit parameters
    :param t: time-stamp of each transit
    :param f: flux sieries
    :param ferr: error sieries
    :param expTime: exposure time
    :returns: likelihood
    :rtype: float

    """
    L = -np.sum((f - transitModel_spectral(params, t, expTime, t01, t02))**2 / (2*(ferr)**2)) -\
        0.5 * np.sum(np.log(2 * np.pi * (ferr)**2))
    return L


def lnprob_spectral(params, params_min, params_max, t, f, ferr, expTime, t01, t02):
    """posterior probability for the transit fit

    :param params: parameter valeus
    :param params_min: upper limit of the parameters
    :param params_max: lower limit of the parameters
    :param t: time stamps for the exposures
    :param f: flux
    :param ferr: flux errors
    :param expTime: exposure times
    :returns: posterior probability
    :rtype: float

    """
    return lnlike_spectral(params, t, f, ferr, expTime, t01, t02) + lnprior(params, params_min, params_max)


def transitModel_white(params, t, expTime):
    """TODO
    a generic transit light curve with batman for broadband light curve

    :param params: transit planet parameters
    :param t: time stamps in seconds
    :param expTime: exposure time for each frame
    :returns: light curve model
    :rtype: np.array

    """
    transit_params.per = 1.51087081  # orbital period
    transit_params.a = 20.4209
    transit_params.inc = 89.65
    transit_params.t0 = params[0]  # time of inferior conjunction
    transit_params.rp = params[1]  # planet radius (in units of stellar radii)
    transit_params.u = [params[4],
                        params[5]]  # linear limb darkening coefficients
    m_b = batman.TransitModel(
        transit_params, (t - 0.5 * expTime) / 86400, exp_time=expTime / 86400)
    lc1 = m_b.light_curve(transit_params)

    transit_params.per = 2.4218233  # orbital period
    transit_params.a = 27.9569
    transit_params.inc = 89.67
    transit_params.t0 = params[2]
    transit_params.rp = params[3]  # planet radius (in units of stellar radii)
    transit_params.u = [params[4], params[5]]  # linear limb darkening
    # coefficients
    m_c = batman.TransitModel(
        transit_params, (t - 0.5 * expTime) / 86400, exp_time=expTime / 86400)
    lc2 = m_c.light_curve(transit_params)
    lc = (lc1 + lc2) - 1  # two transit, remove one baseline

    return lc


def lnlike_white(params, t, f, ferr, expTime):
    """transit profile likelihood function for broadband light curve

    :param params: transit parameters
    :param t: time-stamp of each transit
    :param f: flux sieries
    :param ferr: error sieries
    :param expTime: exposure time
    :returns: likelihood
    :rtype: float

    """
    L = -np.sum((f - transitModel_white(params, t, expTime))**2 / (2*(ferr)**2)) -\
        0.5 * np.sum(np.log(2 * np.pi * (ferr)**2))
    return L


def lnprob_white(params, params_min, params_max, t, f, ferr, expTime):
    """posterior probability for the transit fit

    :param params: parameter valeus
    :param params_min: upper limit of the parameters
    :param params_max: lower limit of the parameters
    :param t: time stamps for the exposures
    :param f: flux
    :param ferr: flux errors
    :param expTime: exposure times
    :returns: posterior probability
    :rtype: float

    """
    return lnlike_white(params, t, f, ferr, expTime) + lnprior(params, params_min, params_max)


def lcBin(lcMat0, lcErrMat0, wavelength0, wavelength):
    """ Bin the light curve matrix to aimed interval

    :param lcMat0: original ramp removed light curve array
    :param lcErrMat0: Uncertainty array for ramp removed array
    :param wavelength0: wavelenght solution for the original matrix
    :param wavelength: wavelength solution for the aimed bin edges
    :returns: the binned light curve matrix
    :rtype: numpy array

    """
    lcLength = lcMat0.shape[1]
    nBin = len(wavelength) - 1  # -1 because wavelegnth represent the edges
    lcBinned = np.zeros((nBin, lcLength))
    errBinned = np.zeros((nBin, lcLength))
    binID = np.digitize(wavelength0, wavelength)
    for i in range(1, 1 + nBin):
        indexBin = np.where(binID == i)[0]
        lcBinned[i-1, :] = lcMat0[indexBin, :].mean(axis=0)
        errBinned[i-1, :] = np.sqrt(np.sum(lcErrMat0[indexBin, :]**2, axis=0)) / len(indexBin)
    return lcBinned, errBinned


def fitTransit(configFileName, plot=False):
    """perform MCMC fit of the transit profiles.

    :param configFileName: configuration file that sets up the light curve fit
    :param plot: whether to create plot during the fit
    :returns: result object
    :rtype: scanData object

    """

    # get configurations as global variables
    conf = configparser.ConfigParser()
    conf.read(configFileName)
    projDIR = path.expanduser(conf['general']['projdir'])
    plotDIR = path.join(projDIR, conf['general']['plotdir'])
    saveDIR = path.join(projDIR, conf['general']['savedir'])
    if not path.exists(plotDIR):
        makedirs(plotDIR)
    if not path.exists(saveDIR):
        makedirs(saveDIR)
    scanDataFN = path.join(projDIR, conf['general']['datafn'])
    # load ramp removed scanning data
    with open(scanDataFN, 'rb') as pkl:
        sd_dict = pickle.load(pkl)
    nWalkers = conf['general'].getint('nwalker')
    nStep = conf['general'].getint('mcmc_nstep')
    burnin = conf['general'].getint('mcmc_nburnin')

    # transit profile parameters
    transit_pNames = json.loads(conf['others']['pnames'])
    for param in transit_pNames:
        transit_params.__setattr__(param, conf['others'][param])

    # fit the broad band light curve first
    t = sd_dict['time']
    wlc = sd_dict['whiteLightCurve_deRamp']
    wlcerr = sd_dict['whiteLightCurveErr_deRamp']
    # normalize light curve to the median
    lc0 = np.median(wlc)
    wlc = wlc / lc0
    wlcerr = wlcerr / lc0
    expTime = sd_dict['expTime']
    LCMat = sd_dict['LCMat_deRamp']
    LCErrMat = sd_dict['LCErrMat_deRamp']
    wavelength0 = sd_dict['wavelength']
    nParams_white = conf['general'].getint('nparams_white')
    pNames_white = json.loads(conf['general']['pnames_white'])

    v0_white = np.zeros(nParams_white)
    vmin_white = np.zeros(nParams_white)
    vmax_white = np.zeros(nParams_white)

    for i, param in enumerate(pNames_white):
        v0_white[i] = conf['white'].getfloat(param)
        vmin, vmax = json.loads(conf['white_limit'][param])
        vmin_white[i] = vmin
        vmax_white[i] = vmax
    nDim_white = len(v0_white)
    pos0_white = [
        v0_white * (1 + 0.001 * np.random.randn(nDim_white)) for k in range(nWalkers)
    ]

    # construct prior and posterior functions
    sampler = emcee.EnsembleSampler(
        nWalkers,
        nDim_white,
        lnprob_white,
        args=(vmin_white, vmax_white, t, wlc, wlcerr, expTime),
        threads=4)
    width = 50
    print('MCMC fit starts!')
    chain, params, param_stds = mcmcRunner(sampler, pos0_white, nStep, burnin, width)
    if plot:
        plt.close('all')
        fig1, fig2 = plotMCMC(
            t,
            wlc,
            wlcerr,
            chain,
            params,
            transitModel_white,
            expTime,
            argName=pNames_white)
        # save results
        fig1.savefig(path.join(plotDIR, 'corner_white.png'))
        fig2.savefig(path.join(plotDIR, 'lc_fit_white.png'))
        plt.close('all')

    with open(path.join(saveDIR, 'mcmc_white.pkl'), 'wb') as pkl:
        pickle.dump({'chain': chain,
                     'params': params,
                     'params_std': param_stds,
                     'params_names': pNames_white}, pkl)
    # use the result from broadband fit in spectral band fit
    t01 = params[0]
    t02 = params[2]
    # fit the spectral band
    nParams_spectral = conf['general'].getint('nparams_spectral')
    pNames_spectral = json.loads(conf['general']['pnames_spectral'])
    v0_spectral = np.zeros(nParams_spectral)
    vmin_spectral = np.zeros(nParams_spectral)
    vmax_spectral = np.zeros(nParams_spectral)

    for i, param in enumerate(pNames_spectral):
        v0_spectral[i] = conf['spectral'].getfloat(param)
        vmin, vmax = json.loads(conf['spectral_limit'][param])
        vmin_spectral[i] = vmin
        vmax_spectral[i] = vmax
    wavelength = json.loads(conf['general']['wavelength'])
    nChannel = len(wavelength) - 1
    LCMat_binned, LCErrMat_binned = lcBin(LCMat, LCErrMat, wavelength0, wavelength)

    Rp1List = np.zeros(nChannel)
    RpErr1List = np.zeros(nChannel)
    Rp2List = np.zeros(nChannel)
    RpErr2List = np.zeros(nChannel)
    for i in range(nChannel):
        lc = LCMat_binned[i, :]
        err = LCErrMat_binned[i, :]
        # normalize
        lc0 = np.median(lc)
        lc = lc / lc0
        err = err / lc0
        pos0_spectral = [
            v0_spectral * (1 + 0.001 * np.random.randn(nParams_spectral))
            for k in range(nWalkers)
        ]
        sampler = emcee.EnsembleSampler(
            nWalkers,
            nParams_spectral,
            lnprob_spectral,
            args=(vmin_spectral, vmax_spectral, t, lc, err, expTime, t01, t02),
            threads=4)
        width = 50
        print('MCMC fit starts!')
        chain, params, param_stds = mcmcRunner(sampler, pos0_spectral, nStep, burnin,
                                               width)
        if plot:
            plt.close('all')
            fig1, fig2 = plotMCMC(
                t,
                lc,
                err,
                chain,
                params,
                transitModel_spectral,
                expTime,
                t01,
                t02,
                argName=pNames_spectral)
            # save results
            fig1.savefig(
                path.join(plotDIR, 'corner_channel_{0:02d}.png'.format(i)))
            fig2.savefig(
                path.join(plotDIR, 'lc_fit_channel_{0:02d}.png'.format(i)))
        with open(path.join(saveDIR, 'mcmc_channel_{0:02d}.pkl'.format(i)), 'wb') as pkl:
            pickle.dump({'chain': chain,
                         'params': params,
                         'params_std': param_stds,
                         'params_names': pNames_spectral}, pkl)

        Rp1List[i] = params[0]
        RpErr1List[i] = param_stds[0]
        Rp2List[i] = params[1]
        RpErr2List[i] = param_stds[1]
    outputDict = {
        'wavelength': wavelength0,
        'Rp1': Rp1List,
        'RpErr1': RpErr1List,
        'Rp2': Rp2List,
        'RpErr2': RpErr2List
    }
    with open(path.join(saveDIR, 'output_spec.pkl'), 'wb') as pkl:
        pickle.dump(outputDict, pkl)
    return


if __name__ == '__main__':
    configFileName = 'FILENAME'  # TODO
    plot = True  # whether to make the plot
    fitTransit(configFileName, plot=plot)
