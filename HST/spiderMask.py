#!/usr/bin/env python3
"""mask out the spider, based on measurement from HD106906
"""
import numpy as np


def spiderMask(centroid, length, width, arraySize=256):
    """calculate the spider position and create a mask for it.
The part where the spiders locate will be marked with NaN
centroid -- center of the PSF
length -- length of the spider as a box radius value (in x)
width -- width of the spider as a radius value"""
    b1 = 1.110
    b2 = -1.104
    x0, y0 = centroid
    mask = np.ones((arraySize, arraySize), dtype=float)
    x = np.arange(x0 - length, x0 + length)
    y1 = b1 * (x - x0) + y0  # y for SWNE spider
    y2 = b2 * (x - x0) + y0  # y for SENW spider
    x = np.around(x).astype(int)
    mask[np.around(y1).astype(int), x] = np.nan
    mask[np.around(y2).astype(int), x] = np.nan
    for w in np.arange(-width, width):
        mask[np.around(y1 + w).astype(int), x] = np.nan
        mask[np.around(y2 + w).astype(int), x] = np.nan
    return mask
