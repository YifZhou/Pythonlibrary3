#!/usr/bin/env python3
"""
mimic the IDL version of xyxy, using astropy wcs
"""

import astropy.wcs as wcs


def xyxy(x, y, hd1, hd2):
    """using all_pic2world and wcs_world2pic to convert the pixels in two images
x, y,
turn out that wcs has different distortions on different images"""
    wcs1 = wcs.WCS(hd1)
    wcs2 = wcs.WCS(hd2)
    ra, dec = wcs1.all_pix2world(x, y, 1)
    x2, y2 = wcs2.all_world2pix(ra, dec, 1)
    return x2, y2

def wcsshift(x0, y0, hd1, hd2):
    """using wcs to measure the shift
"""
    wcs1 = wcs.WCS(hd1)
    wcs2 = wcs.WCS(hd2)
    ra, dec = wcs1.all_pix2world(x0, y0, 1)
    x2, y2 = wcs2.all_world2pix(ra, dec, 1)
    return x2 - x0, y2 - y0

if __name__ == '__main__':
    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt
    fn1 = 'icytd0whq_flt.fits'
    fn2 = 'icytd1xhq_flt.fits'
    hd1 = fits.getheader(fn1, 'sci')
    hd2 = fits.getheader(fn2, 'sci')
    x1 = 138.0
    y1 = 138.0
    x2, y2 = xyxy(x1, y1, hd1, hd2)
    print(x2 - x1, y2 - y1)
