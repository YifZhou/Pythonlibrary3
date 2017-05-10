#! /usr/bin/env python3
"""
read tinytim PSFs
"""
from astropy.io import fits
from scipy.ndimage.interpolation import shift
from os import path
from image import rebin
import numpy as np
from scipy.signal import convolve2d


def downSamp(psf0, sup=9):
    downshape = (psf0.shape[0] // sup, psf0.shape[1] // sup)
    beginPix = (psf0.shape[0] % sup // 2, psf0.shape[1] % sup // 2)
    psf = rebin(psf0[beginPix[0]:beginPix[0]+downshape[0]*sup,
                      beginPix[1]:beginPix[1]+downshape[1]*sup],
                 factor=sup,
                 func=np.sum)
    kernel = np.array([[0.0007, 0.025005,  0.0007],
                       [0.025005,  0.89718,  0.025005],
                       [0.0007, 0.025005, 0.0007]])
    return convolve2d(psf, kernel, mode='same')


def readPSF(fn, DIR='.', shifts=(0, 0), sup=9, downsamp=True):
    fn = path.join(DIR, fn)
    psf0 = fits.getdata(fn)
    psf0 = psf0 - psf0.min() + 1e-13   # remove nagative pixel introduced by numerical error
    psf0 = shift(psf0, (shifts[0]*sup, shifts[1]*sup))
    if not downsamp:
        return psf0
    else:
        return downSamp(psf0, sup)
