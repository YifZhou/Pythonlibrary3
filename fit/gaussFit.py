#! /usr/bin/env python3

from lmfit.models import GaussianModel
"""
convienience function to fit a gaussian
"""


def gaussFit(x, y, dy=None, p0=None):
    """
    x, y : variabile and measurement
    p0: inital guess for a, x0, sigma
    """
    mod = GaussianModel()
    pars = mod.guess(y, x=x)
    if p0 is not None:
        pars['amplitude'].value = p0[0]
        pars['center'].value = p0[1]
        pars['sigma'].value = p0[2]
    if dy is not None:
        weights = 1 / (dy * dy)
    else:
        weights = None
    bestfit = mod.fit(y, x=x, params=pars, weights=weights)
    return bestfit
