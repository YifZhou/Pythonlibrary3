#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy import visualization
from astropy.visualization.mpl_normalize import ImageNormalize

__all__ = ['imshow']


def imshow(im,
           vmin=None,
           vmax=None,
           stretch='asinh',
           clip=False,
           ax=None,
           cmap='viridis',
           interpolation='nearest',
           plotColorbar=True,
           cbarLabel="",
           percentile=0.995):
    """my own imshow function

    :param im: the image array
    :param vmin: minimum value
    :param vmax: maximum value
    :param stretch: stretching method, options: 'asinh', 'log', 'linear'
    :param clip: (default False) flag used in normalization
    :param ax: (default none) axes to plot to
    :param cmap: (default 'viridis') color map to use
    :param interpolation: (default 'nearest') interpolation method to use. Options: 'nearest', 'bilinear', 'bicubic'
    :param plotColorbar: (default True) whether to plot color bar
    :param percentile: (default 0.995) percentile range of values to set the vmin and vmax

    """
    if vmin is None:
        vmin = np.nanpercentile(im, ((1 - percentile) / 2) * 100)
    if vmax is None:
        vmax = np.nanpercentile(im, ((1 - percentile) / 2 + percentile) * 100)
    if stretch == 'asinh':
        stretcher = visualization.AsinhStretch()
    elif stretch == 'log':
        stretcher = visualization.LogStretch()
    else:
        # if stretch method is not specified, use linear
        stretcher = visualization.LinearStretch()
    normalizer = ImageNormalize(vmin=vmin,
                                vmax=vmax,
                                stretch=stretcher,
                                clip=clip)

    if ax is None:
        fig, ax = plt.subplots()
    cax = ax.matshow(im,
                     norm=normalizer,
                     origin='lower',
                     cmap=cmap,
                     interpolation=interpolation)
    if plotColorbar:
        cbar = ax.figure.colorbar(cax, spacing='uniform')
        cbar.ax.set_ylabel(cbarLabel)
    return cax


if __name__ == '__main__':
    from astropy.io import fits
    im = fits.getdata('ibxy01akq_flt.fits', 1)
    cax = imshow(im, vmin=-50, vmax=50, stretch='linear')
    plt.show()
