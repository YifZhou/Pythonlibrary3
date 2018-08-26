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
    y1 = np.around(y1).astype(int)
    y2 = np.around(y2).astype(int)
    index1 = np.where((y1 > 0) & (y1 < shape[0]) &
                      (x > 0) & (x < shape[0]))[0]
    index2 = np.where((y2 > 0) & (y2 < shape[0]) &
                      (x > 0) & (x < shape[0]))[0]
    mask[y1[index1], x[index1]] = 1
    mask[y2[index2], x[index2]] = 1
    for w in np.arange(-width, width):
        y1w = np.around(y1 + w).astype(int)
        y2w = np.around(y2 + w).astype(int)
        index1 = np.where((y1w > 0) & (y1w < shape[0]) &
                          (x > 0) & (x < shape[0]))[0]
        index2 = np.where((y2w > 0) & (y2w < shape[0]) &
                          (x > 0) & (x < shape[0]))[0]
        mask[y1w[index1], x[index1]] = 1
        mask[y2w[index2], x[index2]] = 1
    return mask.astype(bool)
