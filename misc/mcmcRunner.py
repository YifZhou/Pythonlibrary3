#! /usr/bin/env python
import sys


def mcmcRunner(sampler, pos0, nStep, burnin, width=50):
    """run emcee

    :param sampler: emcee sampler
    :param pos0: initial positions of parameters
    :param nStep: step of MCMC sampling
    :param burnin: step of burnin
    :param width: width of 1 step in the progress bar
    :returns: posterior distributions, best values, standard deviation
    :rtype: tuple

    """
    # run MCMC and show the progress bar
    for k, result in enumerate(sampler.sample(pos0, iterations=nStep)):
        n = int((width + 1) * float(k) / nStep)
        sys.stdout.write("\r[{0}{1}] {2}/{3}".format('#' * n, ' ' * (
            width - n), k, nStep))
        sys.stdout.write("\n")

    # calculate the posterior distributions
    nDim = sampler.chain.shape[-1]
    chain = sampler.chain[:, burnin:, :].reshape((-1, nDim))
    # calculate the parameter and standard deviations
    params = chain.mean(axis=0)
    params_std = chain.std(axis=0)
    return chain, params, params_std
