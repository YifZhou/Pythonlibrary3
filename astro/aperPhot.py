# /usr/bin/env python3
"""simple aperture photometry"""
import numpy as np
from astropy.stats import sigma_clip

def aperPhot(image, centroid, starR, skyR1, skyR2, expTime,
             errArray=None, skyMask=None):
    """do simple aperture photometry by adding up the total flux within
the certain aperture

    image -- 2D array, unit in counts per second
    centroid -- the centroid of the star
    starR -- aperture radius used to calculate the star flux
    skyR1 -- inner radius for sky annulus
    skyR2 -- outer radius for sky annulus
    errArray (None) -- uncertaity array for each pixels for error estimation.
            If is None, assuming simple poisson errors. If not None,
            the unit should be the same as iamge
    skyMask (None) -- Mask to exclude certain pixels for sky level
            calculation, e.g., bad pixels, bright starts, spiders. If is None,
            all pixels are used

    return aperture photometry and error estimation,
        sky and sky error estimation"""
    x0, y0 = centroid
    if skyMask is None:
        skyMask = np.ones_like(image)
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    distance_sq = (yy - y0)**2 + (xx - x0)**2
    skys = (image * skyMask)[(distance_sq < skyR2**2) &
                             (distance_sq > skyR1**2)]
    # sigma clip on the sky
    skys = sigma_clip(skys, sigma=3.5, iters=5)
    sky = np.nanmean(skys)

    sky_err = np.nanstd(skys, ddof=1) / np.sqrt(
        skys.count())
    starPixel_i, starPixel_j = np.where(distance_sq < starR**2)
    f = np.sum((image - sky)[starPixel_i, starPixel_j])
    # error calculation, if the error array is not provided, assuming poisson error
    if errArray is None:
        f_err = np.sqrt(f * expTime) / expTime
    # other wise, use the combination of error array and sky error
    else:
        f_err = np.sqrt(
            np.sum(errArray[starPixel_i, starPixel_j]**2 + sky_err**2))
    return f, f_err, sky, sky_err
