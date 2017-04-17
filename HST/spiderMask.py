#!/usr/bin/env python3
"""mask out the spider, based on measurement from HD106906
"""
import numpy as np


def spiderMask(centroid, length, width, shape=(256, 256), qMask=[1, 1, 1, 1]):
    """calculate the spider position and create a mask for it.
The part where the spiders locate will be marked with NaN
centroid -- center of the PSF
length -- length of the spider as a box radius value (in x)
width -- width of the spider as a radius value
qMask -- indicante if all 4 spiders are masked"""
    b1 = 1.110
    b2 = -1.104
    x0, y0 = centroid
    mask = np.zeros(shape, dtype=int)
    x = np.arange(x0 - length, x0 + length)
    y1 = b1 * (x - x0) + y0  # y for SWNE spider
    y2 = b2 * (x - x0) + y0  # y for SENW spider
    x = np.around(x).astype(int)
    mask[np.around(y1).astype(int), x] = 1
    mask[np.around(y2).astype(int), x] = 1
    for w in np.arange(-width, width):
        mask[np.around(y1 + w).astype(int), x] = 1
        mask[np.around(y2 + w).astype(int), x] = 1
    return mask.astype(bool)
