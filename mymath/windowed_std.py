"""Fast method for calculating windowed standard deviation
copied from StackOverflow answer:
https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
"""
# import numpy as np
# from scipy.ndimage.filters import uniform_filter


# def windowed_std(arr, radius):
#     radius = np.array(radius)
#     c1 = uniform_filter(arr, radius * 2, mode='constant', origin=-radius)
#     c2 = uniform_filter(arr * arr, radius * 2, mode='constant', origin=-radius)
#     return ((c2 - c1 * c1)**.5)

from astropy.convolution import convolve, Box2DKernel, Box1DKernel


def windowed2D_std(arr, width, mask=None):
    c1 = convolve(arr, Box2DKernel(width), mask=mask)
    c2 = convolve(arr*arr, Box2DKernel(width), mask=mask)
    return (c2 - c1 * c1)**0.5


def windowed1D_std(arr, width, mask=None):
    c1 = convolve(arr, Box1DKernel(width), mask=mask)
    c2 = convolve(arr*arr, Box1DKernel(width), mask=mask)
    return (c2 - c1 * c1)**0.5
