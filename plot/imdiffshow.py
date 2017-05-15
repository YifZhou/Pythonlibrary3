#!/usr/bin/env python

"""show subtracted image
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

__all__ = ['imdiffshow']


def imdiffshow(im,
               vmin=None,
               vmax=None,
               linthresh=1.0,
               linscale=1.0,
               clip=False,
               ax=None,
               cmap='coolwarm',
               interpolation='nearest',
               plotColorbar=True,
               percentile=0.995):
    """priorily using asinh to strech the image
    """
    if vmin is None:
        vmin = np.nanpercentile(im, ((1 - percentile) / 2) * 100)
    if vmax is None:
        vmax = np.nanpercentile(im, ((1 - percentile) / 2 + percentile) * 100)
    v = max(abs(vmin), abs(vmax))
    vmax = v
    vmin = -v
    normalizer = SymLogNorm(linthresh, linscale, vmin=vmin, vmax=vmax, clip=clip)
    if ax is None:
        fig, ax = plt.subplots()
    cax = ax.matshow(im,
                     norm=normalizer,
                     origin='lower',
                     cmap=cmap,
                     interpolation=interpolation)
    if plotColorbar:
        ax.figure.colorbar(cax, spacing='uniform')
    return cax


if __name__ == '__main__':
    from astropy.io import fits
    im = fits.getdata('ibxy01akq_flt.fits', 1)
    cax = imshow(im, vmin=-50, vmax=50, stretch='linear')
    plt.show()
