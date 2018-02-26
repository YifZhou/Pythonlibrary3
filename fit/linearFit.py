#! /usr/bin/env python
from lmfit.models import LinearModel


def linearFit(x, y, dy=None, p0=None):
    """ convenient function for linear fit using lmfit module

    :param x: x parameter
    :param y: y parameter
    :param dy: uncertainties of y, default is None
    :param p0: initial guess for the fitting parameters

    """
    mod = LinearModel()
    pars = mod.guess(y, x=x)
    # if initial guesses are provided, over-write the auto-guessed result
    if p0 is not None:
        pars['intercept'].value = p0[0]
        pars['slope'].value = p0[1]
    if dy is not None:
        weights = 1 / (dy * dy)
    else:
        weights = None
    bestfit = mod.fit(y, x=x, params=pars, weights=weights)
    return bestfit
