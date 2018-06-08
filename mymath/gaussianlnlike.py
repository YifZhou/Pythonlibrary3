#! /usr/bin/env
"""calculate ln(likelihood) assuming Normal distribution for the residuals
"""
import numpy as np


def gaussianlnlike(obs, mod, sigma=None, N=None):
    """calculate log likelihood assuming normal distribution of the residuals

    :param obs: obsereved values
    :param mod: modeled values, same shape as obs
    :param sigma: observation uncertainties, if None, assume sigma to be 1
    :param N: number of data point, if it is None, assume N=len(obs)
    :returns: log likelihood
    :rtype: float

    """
    if sigma is None:
        sigma = np.ones_like(obs, dtype=float)
    if N is None:
        N = len(obs)
    res = obs - mod
    term1 = -N / 2 * np.log(np.pi)
    term2 = -np.sum(np.log(sigma))
    term3 = -np.sum(res**2 / (2 * sigma**2))
    return term1 + term2 + term3
