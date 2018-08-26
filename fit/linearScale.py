#! /usr/local/env python
"""
get the scale for to minimize form
\[
\sum (y - mx - b)^2 / \sigma^2
\]
"""
import numpy as np


def linearScale(x, y, sigma, fix_bzero=False):
    """
    get the scale for to minimize form
    \[
    \sum (y - mx - b)^2 / \sigma^2
    \]
    """
    s = np.sum(1 / sigma**2)
    sx = np.sum(x / sigma**2)
    sxx = np.sum(x**2 / sigma**2)
    sy = np.sum(y / sigma**2)
    sxy = np.sum(x * y / sigma**2)
    if fix_bzero:
        m = sxy / sxx
        b = 0
    else:
        m = (s * sxy - sx * sy) / (s * sxx - sx**2)
        b = (sy - m * sx) / s
    return m, b
