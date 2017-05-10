#! /usr/bin/env python3

from lmfit.models import GaussianModel
"""
convienience function to fit a gaussian
"""


def gaussFit(x, y, p0=None):
    """
    x, y : variabile and measurement
    p0: inital guess for a, x0, sigma
    """
    mod = GaussianModel()
    pars = mod.guess(y, x=x)
    if p0 is not None:
        pars['amplitude'].value = pars[0]
        pars['center'].value = pars[1]
        pars['sigma'].value = pars[2]
    bestfit = mod.fit(y, x=x)
    return bestfit
