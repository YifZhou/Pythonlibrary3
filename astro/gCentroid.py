#! /usr/bin/env python3
"""
use simple 2D gaussian to fit image centroid
"""
import numpy as np
from functions import gaussian2d
from lmfit import Model, Parameters


gModel = Model(gaussian2d, independent_vars=['x', 'y'])


def gCentroid(x0, y0, im, radius=10, weight=None, mask=None):
    """
    fit a 2d gaussian to the region of interest and return the fitted x and y
    """
    subim = im[round(y0)-radius:round(y0)+radius,
               round(x0)-radius:round(x0)+radius]
    yy, xx = np.meshgrid(np.arange(round(y0) - radius, round(y0) + radius),
                         np.arange(round(x0) - radius, round(x0) + radius),
                         indexing='ij')
    if weight is None:
        weight = np.ones_like(subim)
    if mask is None:
        mask = np.zeros_like(subim)
    else:
        # if there are masked pixel, set the corresponding pixels' weight to 0
        weight[np.where(mask) != 0] = 0
    p = Parameters()
    p.add('x0', value=x0)
    p.add('y0', value=y0)
    p.add('amp', value=np.max(subim))
    p.add('sigma_x', value=1, min=0)
    p.add('sigma_y', value=1, min=0)
    pFit = gModel.fit(subim, params=p, x=xx, y=yy, weights=weight, verbose=True)
    return pFit.best_values['x0'], pFit.best_values['y0']


if __name__ == '__main__':
    from astropy.io import fits
    im = fits.getdata('idde04leq_ima.fits', ('sci', 1))
    cen = gCentroid(31, 144, im)
