#! /usr/bin/env python
import sys
import numpy as np

def mcmcRunner(sampler, pos0, nSteps, nBurnin, width=50):
    """run emcee

    :param sampler: emcee sampler
    :param pos0: initial positions of parameters
    :param nSteps: step of MCMC sampling
    :param nBurnin: step of burnin
    :param width: width of 1 step in the progress bar
    :returns: posterior distributions, best values, standard deviation
    :rtype: tuple

    """
    # run MCMC and show the progress bar
    # run burnin steps first
    print('Running first burnin steps ...')
    p0, lp, _ = sampler.run_mcmc(pos0, nBurnin)
    p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(sampler.k, sampler.dim)
    sampler.reset()
    print('Running second burnin steps ...')
    p0, _, _ = sampler.run_mcmc(pos0, nBurnin)
    sampler.reset()
    print('Running MCMC production')
    for k, result in enumerate(sampler.sample(p0, iterations=nSteps)):
        n = int((width + 1) * float(k) / nSteps)
        sys.stdout.write("\r[{0}{1}] {2}/{3}".format('#' * n, ' ' * (
            width - n), k, nSteps))
        sys.stdout.write("\n")

    # calculate the posterior distributions
    nDim = sampler.chain.shape[-1]
    chain = sampler.chain[:, :, :].reshape((-1, nDim))
    # calculate the parameter and standard deviations
    params = chain.mean(axis=0)
    params_std = chain.std(axis=0)
    return chain, params, params_std
