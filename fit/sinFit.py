#! /usr/bin/env python3
"""
convienience function to fit a single sine wave
"""
import numpy as np
from lmfit import Model


def sin(t, amplitude, period, phase, baseline):
    """sin function to describe the light curve
    """
    return amplitude * np.sin(t/period * np.pi * 2 + phase) + baseline


def sinFit(t, y, dy=None, p0=None):
    """
    t, y : variabile and measurement
    p0: inital guess for a, x0, sigma
    """
    mod = Model(sin)
    if p0 is None:
        p0 = mod.make_params()
    if dy is not None:
        weights = 1 / (dy * dy)
    else:
        weights = None
    bestfit = mod.fit(y, t=t, params=p0, weights=weights)
    return bestfit
